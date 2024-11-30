import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

COLUMNS_TO_DROP = [
    "piezo_station_department_name",
    "piezo_station_commune_code_insee",
    "piezo_station_bss_code",
    "piezo_station_commune_name",
    "piezo_bss_code",
    "piezo_continuity_name",
    "piezo_producer_code",
    "piezo_producer_name",
    "piezo_measure_nature_name",
    "meteo_longitude",
    "meteo_latitude",
    "hydro_observation_date_elab",
    "hydro_status_label",
    "hydro_method_label",
    "hydro_qualification_label",
    "hydro_longitude",
    "hydro_latitude",
    "prelev_longitude_0",
    "prelev_latitude_0",
    "prelev_commune_code_insee_0",
    "prelev_longitude_1",
    "prelev_latitude_1",
    "prelev_commune_code_insee_1",
    "prelev_longitude_2",
    "prelev_latitude_2",
    "prelev_commune_code_insee_2",
    "prelev_structure_code_2",
    "prelev_structure_code_1",
    "prelev_structure_code_0",
    "row_index",
]

columns_to_transform = [
    "piezo_station_pe_label",
    "piezo_station_bdlisa_codes",
    "piezo_station_bss_id",
    "meteo_name",
    "hydro_station_code",
    "hydro_hydro_quantity_elab",
    "prelev_usage_label_0",
    "prelev_usage_label_1",
    "prelev_usage_label_2",
    "region",
]


def complete_nan_meteo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Completes NaN values in columns containing 'meteo' in their names.
    - Numeric columns are filled with the median value grouped by 'piezo_station_department_code'.
    - Categorical columns are filled with the mode value grouped by 'piezo_station_department_code'.

    Args:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with NaN values filled for 'meteo' columns.
    """
    # Identify columns related to "meteo"
    meteo_columns = [col for col in df.columns if "meteo" in col]

    # Separate numeric and categorical "meteo" columns
    numeric_meteo_columns = (
        df[meteo_columns].select_dtypes(include=["float64", "int64"]).columns
    )
    categorical_meteo_columns = (
        df[meteo_columns].select_dtypes(include=["object", "category"]).columns
    )
    # Group by department name
    grouped = df.groupby("piezo_station_department_code")

    # Fill NaN in numeric columns with median
    for col in numeric_meteo_columns:
        df[col] = grouped[col].transform(lambda x: x.fillna(x.median()))

    # Fill NaN in categorical columns with mode
    for col in categorical_meteo_columns:
        df[col] = grouped[col].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else None)
        )

    return df


def complete_nan_prev(df: pd.DataFrame) -> pd.DataFrame:
    """
    Completes NaN values in columns containing 'prelev' in their names.
    - Numeric columns are filled with the median value grouped by 'piezo_station_department_code'.
    - Categorical columns are filled with the mode value grouped by 'piezo_station_department_code'.

    Args:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with NaN values filled for 'prelev' columns.
    """
    # Identify columns related to "prelev"
    prelev_columns = [col for col in df.columns if "prelev" in col]
    print(prelev_columns)
    # Separate numeric and categorical "prelv" columns
    numeric_prelev_columns = (
        df[prelev_columns].select_dtypes(include=["float64", "int64"]).columns
    )
    categorical_prelev_columns = (
        df[prelev_columns].select_dtypes(include=["object", "category"]).columns
    )
    # Group by department name
    grouped = df.groupby("piezo_station_department_code")

    # Fill NaN in numeric columns with median
    for col in numeric_prelev_columns:
        df[col] = grouped[col].transform(lambda x: x.fillna(x.median()))

    # Fill NaN in categorical columns with mode
    for col in categorical_prelev_columns:
        df[col] = grouped[col].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown")
        )

    return df


# Function to map department codes to regions
def map_department_to_region(department_code: str) -> str:

    department_to_region = {
        # Auvergne-Rhône-Alpes
        "01": "Auvergne-Rhône-Alpes",
        "03": "Auvergne-Rhône-Alpes",
        "07": "Auvergne-Rhône-Alpes",
        "15": "Auvergne-Rhône-Alpes",
        "26": "Auvergne-Rhône-Alpes",
        "38": "Auvergne-Rhône-Alpes",
        "42": "Auvergne-Rhône-Alpes",
        "43": "Auvergne-Rhône-Alpes",
        "63": "Auvergne-Rhône-Alpes",
        "69": "Auvergne-Rhône-Alpes",
        "73": "Auvergne-Rhône-Alpes",
        "74": "Auvergne-Rhône-Alpes",
        # Bourgogne-Franche-Comté
        "21": "Bourgogne-Franche-Comté",
        "25": "Bourgogne-Franche-Comté",
        "39": "Bourgogne-Franche-Comté",
        "58": "Bourgogne-Franche-Comté",
        "70": "Bourgogne-Franche-Comté",
        "71": "Bourgogne-Franche-Comté",
        "89": "Bourgogne-Franche-Comté",
        "90": "Bourgogne-Franche-Comté",
        # Bretagne
        "22": "Bretagne",
        "29": "Bretagne",
        "35": "Bretagne",
        "56": "Bretagne",
        # Centre-Val de Loire
        "18": "Centre-Val de Loire",
        "28": "Centre-Val de Loire",
        "36": "Centre-Val de Loire",
        "37": "Centre-Val de Loire",
        "41": "Centre-Val de Loire",
        "45": "Centre-Val de Loire",
        # Corse
        "2A": "Corse",
        "2B": "Corse",
        # Grand Est
        "08": "Grand Est",
        "10": "Grand Est",
        "51": "Grand Est",
        "52": "Grand Est",
        "54": "Grand Est",
        "55": "Grand Est",
        "57": "Grand Est",
        "67": "Grand Est",
        "68": "Grand Est",
        "88": "Grand Est",
        # Hauts-de-France
        "02": "Hauts-de-France",
        "59": "Hauts-de-France",
        "60": "Hauts-de-France",
        "62": "Hauts-de-France",
        "80": "Hauts-de-France",
        # Île-de-France
        "75": "Île-de-France",
        "77": "Île-de-France",
        "78": "Île-de-France",
        "91": "Île-de-France",
        "92": "Île-de-France",
        "93": "Île-de-France",
        "94": "Île-de-France",
        "95": "Île-de-France",
        # Normandie
        "14": "Normandie",
        "27": "Normandie",
        "50": "Normandie",
        "61": "Normandie",
        "76": "Normandie",
        # Nouvelle-Aquitaine
        "16": "Nouvelle-Aquitaine",
        "17": "Nouvelle-Aquitaine",
        "19": "Nouvelle-Aquitaine",
        "23": "Nouvelle-Aquitaine",
        "24": "Nouvelle-Aquitaine",
        "33": "Nouvelle-Aquitaine",
        "40": "Nouvelle-Aquitaine",
        "47": "Nouvelle-Aquitaine",
        "64": "Nouvelle-Aquitaine",
        "79": "Nouvelle-Aquitaine",
        "86": "Nouvelle-Aquitaine",
        "87": "Nouvelle-Aquitaine",
        # Occitanie
        "09": "Occitanie",
        "11": "Occitanie",
        "12": "Occitanie",
        "30": "Occitanie",
        "31": "Occitanie",
        "32": "Occitanie",
        "34": "Occitanie",
        "46": "Occitanie",
        "48": "Occitanie",
        "65": "Occitanie",
        "66": "Occitanie",
        "81": "Occitanie",
        "82": "Occitanie",
        # Pays de la Loire
        "44": "Pays de la Loire",
        "49": "Pays de la Loire",
        "53": "Pays de la Loire",
        "72": "Pays de la Loire",
        "85": "Pays de la Loire",
        # Provence-Alpes-Côte d'Azur
        "04": "Provence-Alpes-Côte d'Azur",
        "05": "Provence-Alpes-Côte d'Azur",
        "06": "Provence-Alpes-Côte d'Azur",
        "13": "Provence-Alpes-Côte d'Azur",
        "83": "Provence-Alpes-Côte d'Azur",
        "84": "Provence-Alpes-Côte d'Azur",
        # French Overseas Regions
        "971": "Guadeloupe",
        "972": "Martinique",
        "973": "Guyane",
        "974": "La Réunion",
        "976": "Mayotte",
    }

    return department_to_region.get(department_code, "Unknown")


def create_region(df: pd.DataFrame) -> pd.DataFrame:
    # Dictionary mapping department codes to regions

    df["region"] = df["piezo_station_department_code"].apply(map_department_to_region)

    return df


def complete_nan_meteo_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Completes NaN values in columns containing 'meteo' in their names.
    - Numeric columns are filled with the median value grouped by 'piezo_station_department_code'.
    - Categorical columns are filled with the mode value grouped by 'piezo_station_department_code'.

    Args:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with NaN values filled for 'meteo' columns.
    """
    # Identify columns related to "meteo"
    meteo_columns = [col for col in df.columns if "meteo" in col]

    # Separate numeric and categorical "meteo" columns
    numeric_meteo_columns = (
        df[meteo_columns].select_dtypes(include=["float64", "int64"]).columns
    )
    categorical_meteo_columns = (
        df[meteo_columns].select_dtypes(include=["object", "category"]).columns
    )
    # Group by department name
    grouped = df.groupby("region")

    # Fill NaN in numeric columns with median
    for col in numeric_meteo_columns:
        df[col] = grouped[col].transform(lambda x: x.fillna(x.median()))

    # Fill NaN in categorical columns with mode
    for col in categorical_meteo_columns:
        df[col] = grouped[col].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else None)
        )

    return df


def complete_nan_national(df: pd.DataFrame) -> pd.DataFrame:

    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns

    # Fill NaN in numeric columns with median
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())

    # Fill NaN in categorical columns with mode
    for col in categorical_columns:
        mode_value = df[col].mode()[0] if not df[col].mode().empty else None
        df[col] = df[col].fillna(mode_value)

    return df


def pre_process_data(x_train):

    x_train = x_train.drop(columns=COLUMNS_TO_DROP)
    threshold = 0.9  # 50% threshold
    columns_to_drop = x_train.columns[x_train.isnull().mean() > threshold]

    x_train_cleaned = x_train.drop(columns=columns_to_drop)
    x_train_completed = complete_nan_meteo(x_train_cleaned)
    x_train_completed_2 = complete_nan_prev(x_train_completed)
    x_train_3 = create_region(x_train_completed_2)
    x_train_4 = complete_nan_meteo_region(x_train_3)
    x_train_5 = complete_nan_national(x_train_4)

    y_train = x_train_5["piezo_groundwater_level_category"]
    y_train_encoded = LabelEncoder().fit_transform(y_train)

    x_train_5 = x_train_5.drop(columns=["piezo_groundwater_level_category"])
    y_train_encoded = pd.DataFrame(y_train_encoded)

    return x_train_5, y_train_encoded


def transform_columns_to_ordinal(df, columns, mappings=None):
    """
    Transforme des colonnes spécifiques en label ordinal tout en conservant le mapping.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les colonnes à transformer.
        columns (list): Liste des colonnes à transformer.
        mappings (dict, optional): Dictionnaire existant des mappings pour chaque colonne.

    Returns:
        pd.DataFrame: Le DataFrame avec les colonnes transformées.
        dict: Dictionnaire des mappings pour réutilisation future.
    """
    # Initialiser les mappings si non fournis
    if mappings is None:
        mappings = {}

    # Encoder chaque colonne spécifiée
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"La colonne '{col}' n'existe pas dans le DataFrame.")

        if col not in mappings:  # Si aucun mapping existant pour cette colonne
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            mappings[col] = {cls: idx for idx, cls in enumerate(le.classes_)}
        else:  # Utiliser un mapping existant
            df[col] = df[col].map(mappings[col]).fillna(-1).astype(int)

    return df, mappings


def ordinal_encode_column(dataframe, column_name, categories_order):
    """
    Encode ordinally a specified column in a dataframe.

    Parameters:
        dataframe (pd.DataFrame): The dataframe containing the column to encode.
        column_name (str): The name of the column to encode.
        categories_order (list): The ordered list of categories.

    Returns:
        pd.DataFrame: The dataframe with the encoded column.
    """
    # Copier le DataFrame pour éviter les modifications inattendues
    dataframe_copy = dataframe.copy()

    # Initialiser l'encodeur ordinal
    encoder = OrdinalEncoder(categories=[categories_order])

    # Encoder la colonne et gérer les types
    dataframe_copy.loc[:, column_name] = encoder.fit_transform(
        dataframe_copy[[column_name]]
    ).astype(int)

    return dataframe_copy


def handle_corse_department(x):
    if isinstance(x, str):
        if x.lower().startswith("corse") or "20" in x:  # Example for identifying Corse
            return 20.0  # Assign a specific value for Corse
        return float(x) if x.isdigit() else 0
    return x  # Return the value as-is if it's not a string


def extract_date_features(df, date_column):
    """
    Extrait les caractéristiques de date (jour, mois, année) d'une colonne et nettoie le DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame contenant la colonne de date.
        date_column (str): Le nom de la colonne de date à traiter.
        drop_original (bool): Indique si la colonne d'origine doit être supprimée (par défaut True).

    Returns:
        pd.DataFrame: Le DataFrame avec les colonnes 'Jour', 'Mois', et 'Année' ajoutées.
    """
    # Convertir la colonne en datetime
    df["datetime"] = pd.to_datetime(df[date_column], errors="coerce")

    # Extraire les informations temporelles
    df["Jour"] = df["datetime"].dt.day.astype("Int64")
    df["Mois"] = df["datetime"].dt.month.astype("Int64")
    df["Année"] = df["datetime"].dt.year.astype("Int64")

    # Supprimer les colonnes d'origine si demandé

    df = df.drop(columns=[date_column, "datetime"])

    return df


def encoded_categorical_features(x_train):

    # Appliquer la transformation avec le cas particulier pour la Corse
    x_train["piezo_station_department_code"] = x_train[
        "piezo_station_department_code"
    ].apply(handle_corse_department)
    x_train["piezo_station_department_code"] = x_train[
        "piezo_station_department_code"
    ].astype(float)
    x_train["insee_%_agri"] = x_train["insee_%_agri"].astype(float)
    x_train["insee_med_living_level"] = x_train["insee_med_living_level"].astype(float)
    x_train["insee_%_ind"] = x_train["insee_%_ind"].astype(float)
    x_train["insee_%_const"] = x_train["insee_%_const"].astype(float)

    x_train = extract_date_features(x_train, "piezo_station_update_date")

    x_train["piezo_measurement_date"] = pd.to_datetime(
        x_train["piezo_measurement_date"]
    )  # Convert to datetime if necessary
    x_train["piezo_measurement_date"] = (
        x_train["piezo_measurement_date"].astype("int64") // 10**9
    )  # Divide by 10^9 to get seconds

    x_train["meteo_date"] = pd.to_datetime(
        x_train["meteo_date"]
    )  # Convert to datetime if necessary
    x_train["meteo_date"] = (
        x_train["meteo_date"].astype("int64") // 10**9
    )  # Divide by 10^9 to get seconds

    x_train, mappings = transform_columns_to_ordinal(x_train, columns_to_transform)

    x_train = ordinal_encode_column(
        x_train,
        "piezo_obtention_mode",
        [
            "Mode d'obtention inconnu",
            "Valeur mesurée",
            "Valeur reconstituée",
            "Valeur corrigée",
        ],
    )
    x_train = ordinal_encode_column(
        x_train,
        "piezo_status",
        [
            "Donnée brute",
            "Donnée contrôlée niveau 1",
            "Donnée contrôlée niveau 2",
            "Donnée interprétée",
        ],
    )
    x_train = ordinal_encode_column(
        x_train,
        "piezo_qualification",
        ["Non Définissable", "Correcte", "Incorrecte", "Incertaine", "Non qualifié"],
    )
    x_train = ordinal_encode_column(
        x_train, "piezo_measure_nature_code", ["N", "I", "D", "S", "0"]
    )
    x_train = ordinal_encode_column(
        x_train,
        "prelev_volume_obtention_mode_label_0",
        [
            "Unknown",
            "Volume forfaitaire",
            "Volume estimé",
            "Volume mesuré",
            "Mesure indirecte",
            "Mesure directe",
        ],
    )
    x_train = ordinal_encode_column(
        x_train,
        "prelev_volume_obtention_mode_label_1",
        [
            "Unknown",
            "Volume forfaitaire",
            "Volume estimé",
            "Volume mesuré",
            "Mesure indirecte",
            "Mesure directe",
        ],
    )
    x_train = ordinal_encode_column(
        x_train,
        "prelev_volume_obtention_mode_label_2",
        [
            "Unknown",
            "Volume forfaitaire",
            "Volume estimé",
            "Volume mesuré",
            "Mesure indirecte",
            "Mesure directe",
        ],
    )

    scaler = StandardScaler()

    # Fit and transform the data
    x_train = scaler.fit_transform(x_train)

    return x_train
