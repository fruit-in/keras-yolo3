from nuscenes.nuscenes import NuScenes

dataroot = 'dataset/nuscenes/'
nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)

sensors = ['CAM_FRONT', 'CAM_BACK',
           'CAM_FRONT_LEFT', 'CAM_BACK_LEFT',
           'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT']
category = {
    'animal': 0,
    'human.pedestrian.adult': 1,
    'human.pedestrian.child': 2,
    'human.pedestrian.construction_worker': 3,
    'human.pedestrian.personal_mobility': 4,
    'human.pedestrian.police_officer': 5,
    'human.pedestrian.stroller': 6,
    'human.pedestrian.wheelchair': 7,
    'movable_object.barrier': 8,
    'movable_object.debris': 9,
    'movable_object.pushable_pullable': 10,
    'movable_object.trafficcone': 11,
    'vehicle.bicycle': 12,
    'vehicle.bus.bendy': 13,
    'vehicle.bus.rigid': 14,
    'vehicle.car': 15,
    'vehicle.construction': 16,
    'vehicle.emergency.ambulance': 17,
    'vehicle.emergency.police': 18,
    'vehicle.motorcycle': 19,
    'vehicle.trailer': 20,
    'vehicle.truck': 21,
    'static_object.bicycle_rack': 22
}

with open('annotations.txt', 'w') as f:
    for scene in nusc.scene:
        sample_token = scene['first_sample_token']

        while sample_token:
            sample = nusc.get('sample', sample_token)

            for sensor in sensors:
                sensor_data = nusc.get('sample_data', sample['data'][sensor])
                if not sensor_data['is_key_frame']:
                    continue

                img_path = dataroot + sensor_data['filename']
                boxes = []

                for ann_token in sample['anns']:
                    for img_ann in nusc.image_annotations:
                        if (img_ann['sample_annotation_token'] == ann_token and
                            img_ann['sample_data_token'] == sensor_data['token'] and
                            img_ann['filename'] == sensor_data['filename']):

                            category_no = category[img_ann['category_name']]
                            box = [int(n) for n in img_ann['bbox_corners']]
                            box.append(category_no)
                            boxes.append(box)

                            break
                if boxes:
                    f.write(img_path)
                    for box in boxes:
                        f.write(' %d,%d,%d,%d,%d' % tuple(box))
                    f.write('\n')

            sample_token = sample['next']
