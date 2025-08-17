### Build Docker Image
#docker build -t california-housing-price-prediction .
#
## sh
## docker run -it california-housing-price-prediction /bin/sh
#
### Run
##docker run -d -p 5001:5001 california-housing-price-prediction
#
#
## tag image
#docker tag california-housing-price-prediction cshekhark/california-housing-price-prediction-repo:latest
#
## push to docker hub
#docker push cshekhark/california-housing-price-prediction-repo:latest


# Build and push image to docker hub
# docker-compose build
# docker-compose push

# Build and Run Docker along with Prometheus
docker-compose up --build
