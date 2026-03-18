import pandas as pd
import numpy as np

def clean_cc_hourly():
    """Clean cc_hourly.csv - handle missing values"""
    df = pd.read_csv('cc_hourly.csv')

    # Fill missing numeric values with 0 for count/payment columns
    numeric_cols = ['PatientCount', 'NewPatientCount', 'ReturnPatientCount',
                    'TotalCash', 'TotalCheque', 'TotalElectronic',
                    'TotalCashPatientCount', 'TotalChequePatientCount',
                    'TotalElectronicPatientCount']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df.to_csv('cc_hourly_cleaned.csv', index=False)
    print(f"cc_hourly.csv: {len(df)} rows cleaned")
    return df

def clean_cc_patient():
    """Clean cc_patient.csv - handle N/A and missing values"""
    df = pd.read_csv('cc_patient.csv')

    # Replace 'N/A' strings with 'Unknown'
    na_replacement_cols = ['Countries/Geography', 'Ethnicities', 'Religions', 'SocialStatus']
    for col in na_replacement_cols:
        if col in df.columns:
            df[col] = df[col].replace(['N/A', '', np.nan], 'Unknown')
            df[col] = df[col].fillna('Unknown')

    # Remove rows where all demographic fields are empty
    df = df.dropna(subset=['ICD-10', 'DiagnosisDescription'], how='all')

    df.to_csv('cc_patient_cleaned.csv', index=False)
    print(f"cc_patient.csv: {len(df)} rows cleaned")
    return df

def clean_cc_clinic_level():
    """Clean cc_clinic_level.csv - fix addresses and state names"""
    df = pd.read_csv('cc_clinic_level.csv')

    # Fix multi-line addresses - replace newlines with spaces
    df['Address'] = df['Address'].str.replace('\n', ' ', regex=False)
    df['Address'] = df['Address'].str.replace(r'\s+', ' ', regex=True).str.strip()

    # Strip whitespace from State column
    df['State'] = df['State'].str.strip()

    # Strip whitespace from District column
    df['District'] = df['District'].str.strip()

    # Ensure numeric columns are proper types
    numeric_cols = ['lat', 'lng', 'TotalRevenue', 'TotalPaid', 'TotalPanel',
                    'TotalPaidPatientCount', 'TotalPanelPatientCount', 'PatientCount',
                    'NewPatientCount', 'ReturnPatientCount', 'TotalCash', 'TotalCheque',
                    'TotalElectronic', 'TotalCashPatientCount', 'TotalChequePatientCount',
                    'TotalElectronicPatientCount']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.to_csv('cc_clinic_level_cleaned.csv', index=False)
    print(f"cc_clinic_level.csv: {len(df)} rows cleaned")
    return df

def clean_cc_doctor():
    """Clean cc_doctor.csv - minimal cleaning needed"""
    df = pd.read_csv('cc_doctor.csv')

    # Strip whitespace from string columns
    string_cols = ['IDOrganisation', 'Address', 'State', 'District', 'StaffName', 'Locum/Residence']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].str.strip()

    df.to_csv('cc_doctor_cleaned.csv', index=False)
    print(f"cc_doctor.csv: {len(df)} rows cleaned")
    return df

def main():
    print("=" * 50)
    print("Data Cleaning Script")
    print("=" * 50)

    clean_cc_hourly()
    clean_cc_patient()
    clean_cc_clinic_level()
    clean_cc_doctor()

    print("=" * 50)
    print("All datasets cleaned! New files created:")
    print("  - cc_hourly_cleaned.csv")
    print("  - cc_patient_cleaned.csv")
    print("  - cc_clinic_level_cleaned.csv")
    print("  - cc_doctor_cleaned.csv")
    print("=" * 50)

if __name__ == "__main__":
    main()
