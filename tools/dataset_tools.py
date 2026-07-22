from pathlib import Path
import pandas as pd

def inspect_dataset(file_path:str) -> dict:
    path=Path(file_path)

    if not path.exists():
        return {
            "success":"False",
            "error": "Dataset does not exist."
        }

    if path.suffix.lower() != ".csv" :
        return {
            "success": "False",
            "error": "Only CSV datasets are currently supported."
        }

    try:
        df = pd.read_csv(path)

        numeric_df = df.select_dtypes(include="number")

        numeric_summary={}

        for column in numeric_df.columns:
            numeric_summary[column]= {
                "mean": float(numeric_df[column].mean())
                if not numeric_df[column].dropna().empty else None,

                "median": float(numeric_df[column].median())
                if not numeric_df[column].dropna().empty else None,

                "min": float(numeric_df[column].min())
                if not numeric_df[column].dropna().empty else None,

                "max": float(numeric_df[column].max())
                if not numeric_df[column].dropna().empty else None
            }

        low_cardinality_columns = []

        for column in df.columns:
            unique_count = df[column].nunique()

            if unique_count <= 20:
                low_cardinality_columns.append({
                    "name": column,
                    "unique_count": int(unique_count),
                    "sample_values": (
                        df[column]
                        .dropna()
                        .unique()[:10]
                        .tolist()
                    )
                })

        return {
            "success": True,
            "file_name": path.name,
            "shape": {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1])
            },
            "columns": [
                {
                    "name": column,
                    "dtype": str(df[column].dtype),
                    "missing": int(df[column].isna().sum()),
                    "unique": int(df[column].nunique())
                }
                for column in df.columns
            ],
            "sample_rows": df.head(5).where(
                pd.notnull(df.head(5)),
                None
            ).to_dict(orient="records"),
            "numeric_summary": numeric_summary,
            "low_cardinality_columns": low_cardinality_columns
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
        