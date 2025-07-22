.PHONY: ml-deploy
ml-deploy:
	cd mal && ./scripts/build_zip.sh
	AWS_ENDPOINT_URL=http://localhost:4566 \
	  S3_BUCKET=greenhouse-ml-artifacts \
	  mal/scripts/publish_artifact.sh build/ml_service.zip ml_service.zip
