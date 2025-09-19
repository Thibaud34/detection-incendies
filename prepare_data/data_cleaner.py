# data_cleaner.py
from pathlib import Path

def get_file_extensions(folder_path: str):
    """
    Retourne la liste des extensions uniques des fichiers dans le dossier donné.

    Args:
        folder_path (str): chemin du dossier

    Returns:
        list: extensions trouvées (ex: ['.jpg', '.png'])
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Le dossier {folder_path} est introuvable.")

    # Compréhension de liste pour récupérer toutes les extensions
    extensions = list({f.suffix.lower() for f in folder.iterdir() if f.is_file()})
    return extensions


