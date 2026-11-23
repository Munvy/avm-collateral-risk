from pathlib import Path
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CSV_PATTERN = (RAW_DIR / "apartments_pl_*.csv").as_posix()
OUTPUT_PATH = PROCESSED_DIR / "apartments_clean.parquet"

TARGET_CITY = "warszawa"


def run_ingest() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    # 1. Wczytanie surowych danych dla Warszawy + wyciągnięcie daty snapshotu z nazwy pliku
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW raw_warsaw AS
        SELECT
            id,
            city,
            COALESCE(NULLIF(TRIM(type), ''), 'unknown') AS type,
            squareMeters AS square_meters,
            rooms,
            floor,
            floorCount AS floor_count,
            buildYear AS build_year,
            latitude,
            longitude,
            centreDistance AS centre_distance_km,
            poiCount AS poi_count,
            schoolDistance AS school_distance_km,
            clinicDistance AS clinic_distance_km,
            postOfficeDistance AS post_office_distance_km,
            kindergartenDistance AS kindergarten_distance_km,
            restaurantDistance AS restaurant_distance_km,
            collegeDistance AS college_distance_km,
            pharmacyDistance AS pharmacy_distance_km,
            COALESCE(NULLIF(TRIM(ownership), ''), 'unknown') AS ownership,
            COALESCE(NULLIF(TRIM(buildingMaterial), ''), 'unknown') AS building_material,
            COALESCE(NULLIF(TRIM(condition), ''), 'unknown') AS condition,
            COALESCE(hasParkingSpace, FALSE) AS has_parking_space,
            COALESCE(hasBalcony, FALSE) AS has_balcony,
            COALESCE(hasElevator, FALSE) AS has_elevator,
            COALESCE(hasSecurity, FALSE) AS has_security,
            COALESCE(hasStorageRoom, FALSE) AS has_storage_room,
            CAST(price AS DOUBLE) AS price,
            CAST(price AS DOUBLE) / NULLIF(squareMeters, 0) AS price_per_sqm,
            regexp_extract(filename, '(\\d{{4}}_\\d{{2}})', 1) AS snapshot_month
        FROM read_csv_auto('{CSV_PATTERN}', filename=true)
        WHERE LOWER(city) = '{TARGET_CITY}'
          AND price > 50000
          AND squareMeters BETWEEN 12 AND 350
          AND latitude IS NOT NULL
          AND longitude IS NOT NULL;
        """
    )

    raw_count = con.execute("SELECT COUNT(*) FROM raw_warsaw;").fetchone()[0]

    # 2. Deduplikacja między miesiącami:
    # Ta sama nieruchomość potrafi wisieć w serwisie ogłoszeniowym przez 2-4 miesiące.
    # Partycjonujemy po zaokrąglonych współrzędnych GPS (~11 m) i parametrach fizycznych lokalu,
    # zachowując NAJNOWSZY snapshot (snapshot_month DESC), aby wyeliminować data leakage
    # między zbiorem treningowym a testowym (Out-of-Time).
    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW deduplicated_warsaw AS
        WITH ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        ROUND(latitude, 4),
                        ROUND(longitude, 4),
                        ROUND(square_meters, 1),
                        COALESCE(rooms, -1),
                        COALESCE(floor, -1),
                        COALESCE(build_year, -1)
                    ORDER BY snapshot_month DESC, id DESC
                ) AS rn
            FROM raw_warsaw
        )
        SELECT * EXCLUDE (rn)
        FROM ranked
        WHERE rn = 1;
        """
    )

    dedup_count = con.execute("SELECT COUNT(*) FROM deduplicated_warsaw;").fetchone()[0]

    # 3. Filtracja skrajnych anomalii cenowych metodą Tukey's IQR na cenie za m2
    # (eliminacja błędów wpisu, np. 1 500 PLN/m2 lub 95 000 PLN/m2 dla luksusowych apartamentów niestandardowych)
    bounds = con.execute(
        """
        SELECT
            QUANTILE_CONT(price_per_sqm, 0.25) AS q1,
            QUANTILE_CONT(price_per_sqm, 0.75) AS q3
        FROM deduplicated_warsaw;
        """
    ).fetchone()

    q1, q3 = bounds[0], bounds[1]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW clean_warsaw AS
        SELECT *
        FROM deduplicated_warsaw
        WHERE price_per_sqm BETWEEN {lower_bound} AND {upper_bound};
        """
    )

    clean_count = con.execute("SELECT COUNT(*) FROM clean_warsaw;").fetchone()[0]

    # 4. Zapis do formatu Parquet
    con.execute(f"COPY clean_warsaw TO '{OUTPUT_PATH.as_posix()}' (FORMAT PARQUET);")

    # 5. Raport kontrolny do weryfikacji
    print("=== PODSUMOWANIE ETAPU 01: INGEST & CLEANING (WARSZAWA) ===")
    print(f"1. Surowe rekordy (po walidacji metrażu/ceny): {raw_count:,}")
    print(f"2. Po deduplikacji (najnowszy snapshot):       {dedup_count:,} (usunięto {raw_count - dedup_count:,} duplikatów)")
    print(f"3. Przedział Tukey IQR dla ceny/m2:            [{lower_bound:,.0f} ; {upper_bound:,.0f}] PLN/m2")
    print(f"4. Finalny czysty zbiór zapisany do Parquet:   {clean_count:,} (odrzucono {dedup_count - clean_count:,} outlierów)")
    print(f"   Ścieżka wyjściowa: {OUTPUT_PATH}")

    print("\n--- ROZKŁAD MIESIĘCZNY (PO CZYSZCZENIU) ---")
    monthly_df = con.execute(
        """
        SELECT
            snapshot_month,
            COUNT(*) AS n_listings,
            ROUND(MEDIAN(price), 0) AS median_price,
            ROUND(MEDIAN(price_per_sqm), 0) AS median_price_sqm,
            ROUND(MEDIAN(square_meters), 1) AS median_sqm
        FROM clean_warsaw
        GROUP BY snapshot_month
        ORDER BY snapshot_month;
        """
    ).df()
    print(monthly_df.to_string(index=False))


if __name__ == "__main__":
    run_ingest()