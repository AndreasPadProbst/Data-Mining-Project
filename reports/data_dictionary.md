# Data Dictionary — Chicago Crimes Dataset

Source: [City of Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2)

## Raw Columns

| Column | Type | Description |
|--------|------|-------------|
| ID | integer | Unique identifier for the record |
| Case Number | string | CPD RD number; unique per incident |
| Date | datetime | Date/time when the incident occurred (may be estimated) |
| Block | string | Partially redacted address |
| IUCR | string | Illinois Uniform Crime Reporting code |
| Primary Type | string | Primary IUCR crime category (e.g. "THEFT", "BATTERY") |
| Description | string | Secondary IUCR description |
| Location Description | string | Location type (e.g. "STREET", "APARTMENT") |
| Arrest | boolean | Whether an arrest was made |
| Domestic | boolean | Whether the incident was domestic-violence related |
| Beat | integer | Police beat |
| District | integer | Police district |
| Ward | integer | City council ward |
| Community Area | integer | Community area number (1–77) |
| FBI Code | string | FBI UCR crime classification code |
| Year | integer | Year of the incident |
| Updated On | datetime | Date the record was last updated |
| Latitude | float | WGS84 latitude |
| Longitude | float | WGS84 longitude |
| Location | string | "(lat, lon)" coordinate string |

## Engineered Columns (Part 1)

| Column | Type | Description |
|--------|------|-------------|
| Community_Area | Int64 | Renamed from "Community Area" |
| Hour | int | Hour of day (0–23) |
| DayOfWeek | int | Day of week (0=Monday, 6=Sunday) |
| DayOfWeekName | str | Full day name |
| Month | int | Month number (1–12) |
| MonthName | str | Full month name |
| YearActual | int | 4-digit year |
| Quarter | int | Quarter (1–4) |
| IsWeekend | bool | True if DayOfWeek in {5, 6} |
| IsHoliday | bool | True if US federal holiday |
| Season | str | Winter/Spring/Summer/Fall |
| LocationGrouped | str | Grouped Location Description bucket |

## Data Quality Notes

- **Mixed-type Location**: Float NaN entries removed; string coordinate entries retained.
- **Missing coordinates**: ~0.3% of rows; dropped.
- **Midnight timestamp spike**: Many records default to 00:00:00 when true time is unknown.
- **IUCR reclassifications**: Some crime codes changed over time; year-over-year comparisons of specific codes should account for this.
