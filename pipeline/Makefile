#!make
.PHONY: all

build:
	docker build . -t $(IMAGE):$(TAG)

train:
	docker run \
		--name ctn-rec-train \
		--rm \
		--volume=$(VOLUME_NAME):/home/rec/data \
		$(IMAGE):$(TAG) \
		python3 train.py \
			--input-reservation-path=$(INPUT_RESERVATION_PATH) \
			--output-model-dir=$(OUTPUT_MODEL_DIR) \
			--lfm-loss=$(LFM_LOSS) \
    		--lfm-num-components=$(LFM_NUM_COMPONENTS)

pack:
	docker run \
		--name ctn-rec-pack-service \
		--rm \
		--volume=$(VOLUME_NAME):/home/rec/data \
		$(IMAGE):$(TAG) \
		python3 pack_service.py \
			--output-model-dir=$(OUTPUT_MODEL_DIR) \
			--service-version=$(TAG)

serve:
	-docker container rm $(CTN_REC_SERVICE_NAME)
	docker run \
		--name $(CTN_REC_SERVICE_NAME) \
		-p 5002:5000 \
		--volume=$(VOLUME_NAME):/home/rec/data \
		$(IMAGE):$(TAG) \
			bentoml serve LightFMRecService:$(TAG)


clean-up:
	docker container rm $(CTN_REC_SERVICE_NAME) \
		&& docker volume rm $(VOLUME_NAME)
