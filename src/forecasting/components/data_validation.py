class DataValidation:

    REQUIRED_COLUMNS = ["State", "Date", "Total"]

    def validate(self, df):

        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)

        if missing:
            raise ValueError(f"Missing columns: {missing}")

        if df.empty:
            raise ValueError("Dataset is empty")

        return True
