IMAGE_VERSION=$(cat DOCKERIMAGEVERSION)
IMAGE_NAME="sckoo/bird-brain:v$IMAGE_VERSION"

docker build -f Dockerfile.gpu -t $IMAGE_NAME-gpu .
docker build -f Dockerfile -t $IMAGE_NAME .
