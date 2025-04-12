from fastapi import UploadFile


class InvalidFile(Exception):
    """Exception, falls die hochgeladene Datei fehlerhaft ist."""

    def __init__(self, file: UploadFile) -> None:
        """Initialisierung von UsernameExistsError mit dem Benutzernamen.

        :param username: Bereits existierender Benutzername
        """
        super().__init__(f"Existierender Benutzername: {username}")
        self.username = username
