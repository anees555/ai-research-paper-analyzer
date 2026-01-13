import os
import secrets
import aiofiles
from fastapi import UploadFile, HTTPException
from app.core.config import settings

class FileService:
    def __init__(self):
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    async def save_upload_file(self, file: UploadFile) -> str:
        """
        Save uploaded file to disk and return absolute path.
        """
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        # Generate unique filename
        filename = f"{secrets.token_hex(8)}_{file.filename}"
        file_path = os.path.join(settings.UPLOAD_DIR, filename)
        abs_path = os.path.abspath(file_path)

        async with aiofiles.open(abs_path, 'wb') as out_file:
            content = await file.read()  # async read
            if len(content) > settings.MAX_UPLOAD_SIZE:
                raise HTTPException(status_code=413, detail="File too large")
            await out_file.write(content)

        return abs_path

file_service = FileService()
