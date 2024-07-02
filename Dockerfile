# Usa una imagen base de Ubuntu Server 20.04 LTS con Python 3.10.1
FROM python:3.10.1-slim

# Evita que la instalación solicite interacción del usuario
ENV DEBIAN_FRONTEND=noninteractive

# Actualiza los repositorios e instala las actualizaciones del sistema
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos de requisitos
COPY requirements.txt .

#Actualziar pip
RUN pip install --upgrade pip

# Instala las dependencias de la aplicación
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de la aplicación al contenedor
COPY . .

# Exponer el puerto en el que correrá la aplicación (ajusta según sea necesario)
EXPOSE 5000

# Comando para correr la aplicación
CMD ["python", "app.py"]
