#curl -X POST http://127.0.0.1:8080/predictions/facedet \
#     -H "Content-Type: application/json" \
#     -d '{"idx": ["img_674ab28f-b75b-41ab-ba60-330a400c4480", "img_c5f95804-175a-4953-9836-34656640499d", "img_ebf522d3-df38-43df-a0eb-fa03addfc3ea"],
#          "lmdb_path" : "/home/victor-ho/work/school/final/backend/myproject/lmdb"
#     }' --output faces.json
#
#
#curl -X POST http://127.0.0.1:8080/predictions/facereg \
#     -H "Content-Type: application/json" \
#     -d @faces.json
curl -X POST http://127.0.0.1:8080/predictions/ocr \
     -H "Content-Type: application/json" \
     -d '{"idx": ["img_c396786e-5e1f-11ef-bf1c-14857f4fde6d"], "lmdb_path": "/home/victor-ho/work/school/final/backend/myproject/lmdb"}'
