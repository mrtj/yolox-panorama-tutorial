FROM public.ecr.aws/panorama/panorama-application
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir opencv-python numpy
COPY src /panorama
