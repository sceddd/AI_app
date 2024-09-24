import numpy as np
from sklearn.metrics import pairwise_distances
from ultralytics import YOLO

#
# item =None
# target = {'face_id': '66d727909626063b57e99d84', 'embedding': [3.361562728881836, 16.33632469177246, 5.4991559982299805]}
#
# temp = [[{'face_id': '66d6ccf28fe0dfc0eac2091a', 'embedding': [3.2458078861236572, 16.258140563964844, 5.323559284210205]}], [{'face_id': '66d725ec9626063b57e99d81', 'embedding': [2.95998477935791, 16.30448341369629, 5.45715856552124]}], [{'face_id': '66d6e2d63dc2c4f2ed6ead54', 'embedding': [2.782545328140259, 16.29608154296875, 5.62630033493042]}], [{'face_id': '66d7de8a5330230718ced45a', 'embedding': [2.91514253616333, 16.309507369995117, 5.487001419067383]}], [{'face_id': '66d7e3995a90b6199ac21ad9', 'embedding': [3.160761594772339, 16.359073638916016, 5.681058406829834]}], [{'face_id': '66d7e3995a90b6199ac21ada', 'embedding': [3.428785562515259, 16.39102554321289, 5.501785755157471]}], [{'face_id': '66d7e04b5a90b6199ac21ad5', 'embedding': [2.8438022136688232, 16.244548797607422, 5.627028942108154]}], [{'face_id': '66d6cd528977d673eaf79131', 'embedding': [3.1296048164367676, 16.31085205078125, 5.54034423828125]}], [{'face_id': '66d6e3253dc2c4f2ed6ead55', 'embedding': [2.722946882247925, 16.14080810546875, 5.681580066680908]}], [{'face_id': '66d7264a9626063b57e99d82', 'embedding': [2.637938976287842, 16.16030502319336, 5.734945774078369]}], [{'face_id': '66d6dc5d8977d673eaf7913f', 'embedding': [3.2091140747070312, 16.30147933959961, 5.6443867683410645]}], [{'face_id': '66d6dc5d8977d673eaf79140', 'embedding': [3.428785562515259, 16.39102554321289, 5.501785755157471]}], [{'face_id': '66d7e3e25a90b6199ac21adb', 'embedding': [3.1279616355895996, 16.234825134277344, 5.244141578674316]}], [{'face_id': '66d6e1a63dc2c4f2ed6ead51', 'embedding': [3.4160866737365723, 16.34581184387207, 5.689699172973633]}], [{'face_id': '66d6e1a63dc2c4f2ed6ead52', 'embedding': [3.471439838409424, 16.34551429748535, 5.469798564910889]}], [{'face_id': '66d6ca328fe0dfc0eac20916', 'embedding': [3.2008702754974365, 16.299819946289062, 5.655550479888916]}], [{'face_id': '66d6ca328fe0dfc0eac20917', 'embedding': [3.41027569770813, 16.393260955810547, 5.5120439529418945]}], [{'face_id': '66d726a99626063b57e99d83', 'embedding': [2.6795923709869385, 16.19782066345215, 5.519704341888428]}], [{'face_id': '66d6db2f8977d673eaf7913e', 'embedding': [2.9652957916259766, 16.3077449798584, 5.463258743286133]}], [{'face_id': '66d6e1f33dc2c4f2ed6ead53', 'embedding': [2.9650354385375977, 16.30772590637207, 5.463475227355957]}], [{'face_id': '66d7e2a85a90b6199ac21ad6', 'embedding': [2.930756092071533, 16.308868408203125, 5.477476119995117]}], [{'face_id': '66d7df585330230718ced45b', 'embedding': [2.924717426300049, 16.30833625793457, 5.477123260498047]}], [{'face_id': '66d7e2e35a90b6199ac21ad7', 'embedding': [2.965196132659912, 16.30774688720703, 5.4633612632751465]}], [{'face_id': '66d7dfd45330230718ced45c', 'embedding': [2.9212677478790283, 16.309865951538086, 5.482792377471924]}], [{'face_id': '66d7de535330230718ced458', 'embedding': [3.169281244277954, 16.358713150024414, 5.690634250640869]}], [{'face_id': '66d7de535330230718ced459', 'embedding': [3.429060697555542, 16.390743255615234, 5.500342845916748]}], [{'face_id': '66d7e35d5a90b6199ac21ad8', 'embedding': [2.965165376663208, 16.307741165161133, 5.463376045227051]}]]
#
#
# def pairwise_find(embeds_list, new_point, k=4):
#     # Chuyển đổi danh sách các embedding và new_point thành mảng NumPy
#     embeds = np.array([e[0]['embedding'] for e in embeds_list])  # Lấy embedding từ danh sách con
#     new_point = np.array(new_point['embedding']).reshape(1, -1)
#
#     # Tính toán khoảng cách giữa new_point và các embedding
#     distances = pairwise_distances(embeds, new_point, metric='euclidean').flatten()
#
#     # Tìm các chỉ số của k embedding gần nhất
#     nearest_indices = np.argsort(distances)[:k]
#
#     # Tạo danh sách các đối tượng gần nhất với embedding và khoảng cách
#     nearest_points = [{'embedding': embeds[i].tolist(), 'distance': distances[i]} for i in nearest_indices]
#
#     # Trả về danh sách các điểm gần nhất và khoảng cách của chúng
#     return nearest_points
#
# nearest_embeddings = pairwise_find(temp, target, k=4)
# print(nearest_embeddings)
model = YOLO("/home/victor-ho/work/school/final/backend/WODex/model/object_detection/weights/yolov8s-world.pt")
model.set_classes(["person","shoes","bag"])
model.predict("/home/victor-ho/work/school/final/backend/WODex/test_img/download.jpeg")