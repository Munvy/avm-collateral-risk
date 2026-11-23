from pathlib import Path
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "apartments_clean.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "apartments_features.parquet"

# Centrum Warszawy (Pałac Kultury i Nauki)
PKIN_LAT = 52.2319
PKIN_LON = 21.0067

# Pełna lista 39 czynnych stacji metra M1 i M2 w Warszawie (2023-2024)
WARSAW_METRO_STATIONS = [
    # Linia M1 (Kabaty -> Młociny)
    ("Kabaty", 52.1314, 21.0653),
    ("Natolin", 52.1400, 21.0575),
    ("Imielin", 52.1498, 21.0454),
    ("Stokłosy", 52.1562, 21.0345),
    ("Ursynów", 52.1620, 21.0276),
    ("Służew", 52.1733, 21.0263),
    ("Wilanowska", 52.1818, 21.0229),
    ("Wierzbno", 52.1901, 21.0168),
    ("Racławicka", 52.1982, 21.0123),
    ("Pole Mokotowskie", 52.2089, 21.0078),
    ("Politechnika", 52.2176, 21.0147),
    ("Centrum", 52.2304, 21.0108),
    ("Świętokrzyska", 52.2354, 21.0083),
    ("Ratusz Arsenał", 52.2448, 21.0013),
    ("Dworzec Gdański", 52.2582, 20.9944),
    ("Plac Wilsona", 52.2691, 20.9849),
    ("Marymont", 52.2718, 20.9716),
    ("Słodowiec", 52.2769, 20.9599),
    ("Stare Bielany", 52.2813, 20.9499),
    ("Wawrzyszew", 52.2864, 20.9395),
    ("Młociny", 52.2909, 20.9293),
    # Linia M2 (Bemowo -> Bródno)
    ("Bemowo", 52.2392, 20.9135),
    ("Ulrychów", 52.2406, 20.9317),
    ("Księcia Janusza", 52.2391, 20.9437),
    ("Młynów", 52.2377, 20.9598),
    ("Płocka", 52.2330, 20.9665),
    ("Rondo Daszyńskiego", 52.2301, 20.9830),
    ("Rondo ONZ", 52.2330, 20.9984),
    ("Nowy Świat-Uniwersytet", 52.2368, 21.0169),
    ("Centrum Nauki Kopernik", 52.2398, 21.0315),
    ("Stadion Narodowy", 52.2468, 21.0431),
    ("Dworzec Wileński", 52.2543, 21.0355),
    ("Szwedzka", 52.2626, 21.0445),
    ("Targówek Mieszkaniowy", 52.2692, 21.0504),
    ("T Trocka", 52.2756, 21.0554),
    ("Zacisze", 52.2836, 21.0623),
    ("Kondratowicza", 52.2921, 21.0493),
    ("Bródno", 52.2935, 21.0294),
]


def build_features() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Brak pliku wejściowego {INPUT_PATH}. Uruchom najpierw 01_ingest.py.")

    con = duckdb.connect()

    # Rejestracja tabeli stacji metra w DuckDB
    con.execute("CREATE OR REPLACE TEMP TABLE metro_stations (station_name VARCHAR, m_lat DOUBLE, m_lon DOUBLE);")
    con.executemany("INSERT INTO metro_stations VALUES (?, ?, ?);", WARSAW_METRO_STATIONS)

    # Zapytanie obliczające cechy geoprzestrzenne (Haversine), strukturalne oraz czasowe
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW featured_apartments AS
        WITH base AS (
            SELECT
                a.*,
                -- Wzór Haversine: odległość w km do Pałacu Kultury i Nauki
                2 * 6371.0 * ASIN(SQRT(
                    POWER(SIN(RADIANS(a.latitude - {PKIN_LAT}) / 2), 2) +
                    COS(RADIANS({PKIN_LAT})) * COS(RADIANS(a.latitude)) *
                    POWER(SIN(RADIANS(a.longitude - {PKIN_LON}) / 2), 2)
                )) AS dist_pkin_km,

                -- Kąt biegunowy względem centrum (pozwala uchwycić kierunek dzielnicy: Mokotów vs Praga vs Bemowo)
                ATAN2(a.latitude - {PKIN_LAT}, a.longitude - {PKIN_LON}) AS angle_from_center_rad
            FROM read_parquet('{INPUT_PATH.as_posix()}') a
        ),
        metro_dist AS (
            SELECT
                b.id,
                MIN(
                    2 * 6371.0 * ASIN(SQRT(
                        POWER(SIN(RADIANS(b.latitude - m.m_lat) / 2), 2) +
                        COS(RADIANS(m.m_lat)) * COS(RADIANS(b.latitude)) *
                        POWER(SIN(RADIANS(b.longitude - m.m_lon) / 2), 2)
                    ))
                ) AS dist_nearest_metro_km
            FROM base b
            CROSS JOIN metro_stations m
            GROUP BY b.id
        ),
        month_mapping AS (
            SELECT
                snapshot_month,
                DENSE_RANK() OVER (ORDER BY snapshot_month) - 1 AS month_idx
            FROM (SELECT DISTINCT snapshot_month FROM base)
        )
        SELECT
            b.*,
            ROUND(md.dist_nearest_metro_km, 3) AS dist_nearest_metro_km,
            CASE WHEN md.dist_nearest_metro_km <= 0.8 THEN 1 ELSE 0 END AS is_walking_distance_metro,

            -- Cechy strukturalne mieszkania i budynku
            ROUND(b.square_meters / NULLIF(b.rooms, 0), 2) AS sqm_per_room,
            ROUND(b.floor / NULLIF(b.floor_count, 0), 3) AS relative_floor,
            CASE WHEN b.floor = 0 THEN 1 ELSE 0 END AS is_ground_floor,
            CASE WHEN b.floor = b.floor_count AND b.floor_count > 1 THEN 1 ELSE 0 END AS is_top_floor,
            CASE WHEN b.build_year IS NOT NULL THEN 2024 - b.build_year ELSE NULL END AS building_age,
            CASE WHEN b.build_year IS NULL THEN 1 ELSE 0 END AS is_build_year_missing,

            -- Syntetyczny wskaźnik udogodnień (0 - 5)
            (
                CAST(b.has_parking_space AS INTEGER) +
                CAST(b.has_balcony AS INTEGER) +
                CAST(b.has_elevator AS INTEGER) +
                CAST(b.has_security AS INTEGER) +
                CAST(b.has_storage_room AS INTEGER)
            ) AS amenities_score,

            -- Wymiar czasowy i reżim rynkowy (Bezpieczny Kredyt 2%)
            mm.month_idx,
            CASE WHEN b.snapshot_month <= '2023_12' THEN 1 ELSE 0 END AS is_bk2_period
        FROM base b
        JOIN metro_dist md ON b.id = md.id
        JOIN month_mapping mm ON b.snapshot_month = mm.snapshot_month;
        """
    )

    con.execute(f"COPY featured_apartments TO '{OUTPUT_PATH.as_posix()}' (FORMAT PARQUET);")

    # Raport kontrolny nowych cech
    print("=== PODSUMOWANIE ETAPU 02: FEATURE ENGINEERING ===")
    stats_df = con.execute(
        """
        SELECT
            COUNT(*) AS total_rows,
            ROUND(MEDIAN(dist_nearest_metro_km), 2) AS median_metro_km,
            ROUND(AVG(is_walking_distance_metro) * 100, 1) AS pct_near_metro_800m,
            ROUND(MEDIAN(sqm_per_room), 1) AS median_sqm_per_room,
            ROUND(MEDIAN(building_age), 0) AS median_building_age,
            ROUND(AVG(amenities_score), 2) AS avg_amenities_score
        FROM featured_apartments;
        """
    ).df()
    print(stats_df.to_string(index=False))

    print("\n--- WERYFIKACJA PREMII ZA METRO (MEDIANA CENY/M2) ---")
    metro_effect_df = con.execute(
        """
        SELECT
            CASE
                WHEN dist_nearest_metro_km <= 0.5 THEN '1. <= 500m od metra'
                WHEN dist_nearest_metro_km <= 1.0 THEN '2. 500m - 1km od metra'
                WHEN dist_nearest_metro_km <= 2.5 THEN '3. 1km - 2.5km od metra'
                ELSE '4. > 2.5km od metra'
            END AS metro_zone,
            COUNT(*) AS n_listings,
            ROUND(MEDIAN(price_per_sqm), 0) AS median_price_sqm,
            ROUND(MEDIAN(dist_pkin_km), 1) AS median_dist_center_km
        FROM featured_apartments
        GROUP BY 1
        ORDER BY 1;
        """
    ).df()
    print(metro_effect_df.to_string(index=False))
    print(f"\nZapisano wzbogacony zbiór do: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_features()