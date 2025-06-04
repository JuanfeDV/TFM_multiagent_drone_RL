import os



def get_latest_checkpoint(checkpoint_dir: str):
    """
    Devuelve la ruta al último archivo de checkpoint (.zip) en un directorio.

    Args:
        checkpoint_dir (str): Ruta al directorio donde se almacenan los checkpoints.

    Returns:
        str | None: Ruta completa al archivo más reciente, o None si no existe ninguno.
    """

    # Se verifica si existe el directorio. En caso contrario, no hay checkpoint
    if not os.path.exists(checkpoint_dir): 
        return None

    # Se obtienen todos los ficheros .zip (checkpoints). En caso de no existir ficheros .zip, no hay checkpoints
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")]
    if not files: 
        return None

    # Se busca el fichero más reciente por fecha de modificación para obtener el último checkpoint
    latest = max(
        files, 
        key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f))
        )


    return os.path.join(checkpoint_dir, latest)

