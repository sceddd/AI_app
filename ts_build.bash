while getopts ":b" opt; do
  case $opt in
    b)
      cd model/face && zip -r util.zip util
      cd ../..
      torch-model-archiver --model-name facedet --version 1.0 --handler ./model/face/det_model.py --export-path ./model/model_store -f
      torch-model-archiver --model-name facereg --version 1.0 --serialized-file ./model/face/weights/vgg_face_dag.pth --handler ./model/face/recog_model.py --extra-files ./model/face/util.zip --export-path ./model/model_store -f
      torch-model-archiver --model-name ocr --version 1.0 --handler ./model/ocr/ocr_model.py --export-path ./model/model_store -f --extra-files ./model/ocr/ocr.py
      torch-model-archiver --model-name obdet --version 1.0 --handler ./model/object_detection/detect_model.py --serialized-file ./model/object_detection/weights/yolov8s-world.pt --export-path ./model/model_store -f
    ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done
torchserve --stop
torchserve --start --foreground --ts-config ./model/configs.properties --models facedet=facedet.mar,facereg=facereg.mar,ocr=ocr.mar,obdet=obdet.mar --model-store ./model/model_store --disable-token-auth