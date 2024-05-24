import pandas as pd
from collections import defaultdict

class FeatureManager:
    def __init__(self):
        self.user_auth_count = defaultdict(int)
        self.computer_auth_count = defaultdict(int)
        self.user_success_count = defaultdict(int)
        self.user_attempt_count = defaultdict(int)
        self.computer_success_count = defaultdict(int)
        self.computer_attempt_count = defaultdict(int)
        self.user_last_login_hour = defaultdict(list)

    def update_global_state(self, df: pd.DataFrame):
        for user in df['source_user@domain']:
            self.user_auth_count[user] += 1
        for computer in df['source_computer']:
            self.computer_auth_count[computer] += 1
        for idx, row in df.iterrows():
            user = row['source_user@domain']
            computer = row['source_computer']
            success = row['success']
            time = row['time']
            self.user_attempt_count[user] += 1
            self.computer_attempt_count[computer] += 1
            self.user_last_login_hour[user].append(time.hour)
            if success:
                self.user_success_count[user] += 1
                self.computer_success_count[computer] += 1

    def handle_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        # Example: Fill nulls with default values or use statistical imputation
        df['source_user@domain'].fillna('unknown_user', inplace=True)
        df['destination_user@domain'].fillna('unknown_user', inplace=True)
        df['source_computer'].fillna('unknown_computer', inplace=True)
        df['destination_computer'].fillna('unknown_computer', inplace=True)
        df['authentication_type'].fillna('unknown', inplace=True)
        df['logon_type'].fillna('unknown', inplace=True)
        df['authentication_orientation'].fillna('unknown', inplace=True)
        df['success'].fillna(0, inplace=True)
        df['domain'].fillna('unknown', inplace=True)
        df['domain_controller'].fillna('unknown', inplace=True)
        df['event_id'].fillna(-1, inplace=True)
        df['process_name'].fillna('unknown_process', inplace=True)
        df['logon_id'].fillna('unknown_logon', inplace=True)
        df['ip_address'].fillna('0.0.0.0', inplace=True)
        df['sub_status'].fillna('unknown', inplace=True)
        df['failure_reason'].fillna('unknown', inplace=True)

        return df
    
    def detect_account_escalation(self, df: pd.DataFrame) -> pd.DataFrame:
        df['escalated_account'] = df.groupby('source_user@domain')['destination_user@domain'].transform(
            lambda x: x.ne(x.shift()).astype(int)
        )
        return df

    def detect_password_spraying(self, df: pd.DataFrame, time_window: int = 60, threshold: int = 5) -> pd.DataFrame:
        df = df.sort_values(by=['time'])
        df['failed_attempts'] = (df['success'] == 0).astype(int)
        df['failed_attempts_window'] = df['failed_attempts'].rolling(f'{time_window}S', on='time').sum()
        df['password_spraying'] = df['failed_attempts_window'] > threshold
        return df

    def detect_brute_force_followed_by_success(self, df: pd.DataFrame, time_window: int = 60) -> pd.DataFrame:
        df = df.sort_values(by=['time'])
        df['failed_attempts'] = (df['success'] == 0).astype(int)
        df['success_attempt'] = (df['success'] == 1).astype(int)
        
        df['failed_before_success'] = df['failed_attempts'].rolling(f'{time_window}S', on='time').sum()
        df['successful_after_failed'] = df['failed_before_success'].shift(-1) > 0
        df['brute_force_success'] = df['success_attempt'] & df['successful_after_failed']
        
        return df

    def detect_unusual_login_times(self, df: pd.DataFrame) -> pd.DataFrame:
        def calculate_unusual_time(user, hour):
            last_hours = self.user_last_login_hour[user]
            if not last_hours:
                return 0
            return int(hour not in last_hours)
        
        df['unusual_login_time'] = df.apply(lambda row: calculate_unusual_time(row['source_user@domain'], row['hour']), axis=1)
        return df
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.handle_nulls(df)
        df['time'] = pd.to_datetime(df['time'])
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['month'] = df['time'].dt.month
        
        df['user_auth_count'] = df['source_user@domain'].map(self.user_auth_count)
        df['computer_auth_count'] = df['source_computer'].map(self.computer_auth_count)
        
        df['user_success_rate'] = df['source_user@domain'].map(
            lambda user: self.user_success_count[user] / self.user_attempt_count[user] if self.user_attempt_count[user] > 0 else 0
        )
        df['computer_success_rate'] = df['source_computer'].map(
            lambda computer: self.computer_success_count[computer] / self.computer_attempt_count[computer] if self.computer_attempt_count[computer] > 0 else 0
        )

        df = self.detect_account_escalation(df)
        df = self.detect_password_spraying(df)
        df = self.detect_brute_force_followed_by_success(df)
        df = self.detect_unusual_login_times(df)

        return df
