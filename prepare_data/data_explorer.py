# Compter le nombre total d'image
def count_images(images_df):
    """
    Retourne le nombre total d'images dans le DataFrame images_df
    """
    return len(images_df)


# Quelles sont les catégories ?
def list_categories(categories_df):
    """
    Retourne la liste des catégories disponibles
    """
    return categories_df["name"].tolist()




# Statistiques sur le nombre d'annotation par image
def annotations_statistics(annotations_df, images_df=None):
    """
    Retourne les stats sur le nombre d'annotations par image.
    Si images_df est fourni, ajoute le nom de fichier.
    """
    stats = annotations_df.groupby("image_id").size().reset_index(name="nb_annotations")
    
    if images_df is not None:
        stats = stats.merge(images_df, left_on="image_id", right_on="id")
    
    return stats
