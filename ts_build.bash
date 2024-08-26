#zip -r util.zip util/
#torch-model-archiver --model-name facedet --version 1.0 --handler det_model.py --export-path model_store -f
#torch-model-archiver --model-name facereg --version 1.0 --serialized-file ./weight/vgg_face_dag.pth --handler recog_model.py --extra-files util.zip   --export-path model_store -f
torchserve --stop
torchserve --start --foreground --ts-config ./face_recognition/face/configs.properties --models facedet=facedet.mar facereg=facereg.mar --model-store ./face_recognition/face/model_store --disable-token-auth
