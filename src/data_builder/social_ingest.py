import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
file_dir = BASE_DIR / ".." / ".." / "data" / "static"

air_fryer = pd.read_csv(file_dir/'reddit - air fryer.csv')
air_purifiers = pd.read_csv(file_dir/'reddit - air purifiers.csv')
appliances = pd.read_csv(file_dir/'reddit - appliances.csv')
robot_vacuums = pd.read_csv(file_dir/'reddit - robot vacuums.csv')
vacuum_cleaners = pd.read_csv(file_dir/'reddit - vacuum cleaners.csv')

air_fryer = air_fryer[['absolute', 'block', 'md', 'topic']].copy()
air_purifiers = air_purifiers[['absolute', 'block', 'md', 'md 2', 'md 3', 'md 4', 'md 7', 'topic']].copy()
appliances = appliances[['absolute', 'block', 'md', 'md 2', 'topic']].copy()
robot_vacuums = robot_vacuums[['absolute', 'block', 'md', 'md 2', 'md 3', 'md 4', 'md 5', 'topic']].copy()
vacuum_cleaners = vacuum_cleaners[['absolute', 'block', 'md', 'md 6', 'md 10', 'topic']].copy()

#--------
air_fryer = air_fryer.astype(str)


#--------
md_cols = ['md', 'md 2', 'md 3', 'md 4', 'md 7']
air_purifiers = (
    air_purifiers
    .astype(str)
    .melt(
        id_vars=['absolute', 'block', 'topic'],
        value_vars=md_cols,
        var_name='md_source',
        value_name='md_value'
    )
    .drop(columns='md_source')
    .rename(columns={'md_value': 'md'})
)

air_purifiers = air_purifiers[
    air_purifiers['md'].str.strip().ne('') &
    air_purifiers['md'].ne('nan')
].reset_index(drop=True)


#--------
appliances = appliances.astype(str)

md_cols = ['md', 'md 2']

appliances = (
    appliances
    .melt(
        id_vars=['absolute', 'block', 'topic'],
        value_vars=md_cols,
        var_name='md_source',
        value_name='md_value'
    )
    .drop(columns='md_source')
    .rename(columns={'md_value': 'md'})
)

appliances = appliances[
    appliances['md'].str.strip().ne('') &
    appliances['md'].ne('nan')
].reset_index(drop=True)


#--------
robot_vacuums = robot_vacuums.astype(str)

md_cols = ['md', 'md 2', 'md 3', 'md 4', 'md 5']

robot_vacuums = (
    robot_vacuums
    .melt(
        id_vars=['absolute', 'block', 'topic'],
        value_vars=md_cols,
        var_name='md_source',
        value_name='md_value'
    )
    .drop(columns='md_source')
    .rename(columns={'md_value': 'md'})
)

robot_vacuums = robot_vacuums[
    robot_vacuums['md'].str.strip().ne('') &
    robot_vacuums['md'].ne('nan')
].reset_index(drop=True)


#--------
vacuum_cleaners = vacuum_cleaners.astype(str)

md_cols = ['md', 'md 6', 'md 10']

vacuum_cleaners = (
    vacuum_cleaners
    .melt(
        id_vars=['absolute', 'block', 'topic'],
        value_vars=md_cols,
        var_name='md_source',
        value_name='md_value'
    )
    .drop(columns='md_source')
    .rename(columns={'md_value': 'md'})
)

vacuum_cleaners = vacuum_cleaners[
    vacuum_cleaners['md'].str.strip().ne('') &
    vacuum_cleaners['md'].ne('nan')
].reset_index(drop=True)

social_df = pd.concat([air_fryer, air_purifiers, vacuum_cleaners, robot_vacuums, appliances])

social_df.to_csv(file_dir / 'raw - socials.csv', index=False)
