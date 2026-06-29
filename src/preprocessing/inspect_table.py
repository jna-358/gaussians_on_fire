import sqlite3
import numpy as np

def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) // 2147483647
    return image_id1, image_id2

def insert_matches(matches: list, db_path: str):
    num_matches = len(matches[0])
    num_images = len(matches)
        
    # Get existing keypoints
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT image_id, rows, cols, data FROM keypoints")
        data_keypoints = cursor.fetchall()

    # Modify the keypoints data
    keypoint_injection = []
    for i, (image_id, rows, cols, data) in enumerate(data_keypoints):
        # Load existing keypoints
        data_array = np.frombuffer(data, dtype=np.float32)
        data_array = data_array.reshape(rows, cols)
        
        # Prepare new matches - COLMAP expects keypoints as (x, y, scale, orientation, response, octave)
        # We have (x, y) so we need to pad with zeros for the other 4 parameters
        new_matches = matches[i].copy()
        
        # Ensure we have the right shape and data type
        if new_matches.shape[1] != 2:
            raise ValueError(f"Expected matches to have shape (N, 2), got {new_matches.shape}")
        
        # Handle NaN values
        new_matches[np.isnan(new_matches)] = 0
        
        # Pad with zeros for the additional COLMAP keypoint parameters
        # COLMAP keypoint format: [x, y, scale, orientation, response, octave]
        padded_matches = np.zeros((new_matches.shape[0], 6), dtype=np.float32)
        padded_matches[:, :2] = new_matches
        
        # Concatenate with existing keypoints
        data_array_new = np.concatenate([data_array, padded_matches], axis=0)
        
        # Ensure data is contiguous and correct type
        data_array_new = np.ascontiguousarray(data_array_new, dtype=np.float32)
        
        keypoint_injection.append((
            image_id, 
            data_array_new.shape[0], 
            data_array_new.shape[1], 
            data_array_new.tobytes()
        ))
        
        print(f"Image {image_id}: {rows} existing keypoints + {new_matches.shape[0]} new = {data_array_new.shape[0]} total")

    # Get existing two view geometries
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT pair_id, rows, cols, data, config, F, E, H, qvec, tvec FROM two_view_geometries")
        data_two_view_geometries = cursor.fetchall()

    two_view_geometry_injection = []
    for pair_id, rows, cols, data, config, F, E, H, qvec, tvec in data_two_view_geometries:
        image_id_1, image_id_2 = pair_id_to_image_ids(pair_id)
        
        # Load existing matches
        data_array = np.frombuffer(data, dtype=np.uint32)
        data_array = data_array.reshape(rows, cols)
        
        # Find new matches between this pair
        matches_new = []
        for i_match in range(num_matches):
            # Check if feature is present in both images
            if (not np.any(np.isnan(matches[image_id_1-1][i_match]))) and (not np.any(np.isnan(matches[image_id_2-1][i_match]))):
                # Calculate keypoint indices (0-based)
                keypoint_id_1 = data_keypoints[image_id_1-1][1] + i_match  # existing + new index
                keypoint_id_2 = data_keypoints[image_id_2-1][1] + i_match
                matches_new.append((keypoint_id_1, keypoint_id_2))

        if matches_new:
            matches_new = np.array(matches_new, dtype=np.uint32)
            data_array_new = np.concatenate([data_array, matches_new], axis=0)
            print(f"Pair {image_id_1}-{image_id_2}: {rows} existing matches + {len(matches_new)} new = {data_array_new.shape[0]} total")
        else:
            data_array_new = data_array
            print(f"Pair {image_id_1}-{image_id_2}: {rows} existing matches + 0 new = {data_array_new.shape[0]} total")
        
        # Ensure data is contiguous and correct type
        data_array_new = np.ascontiguousarray(data_array_new, dtype=np.uint32)
        
        two_view_geometry_injection.append((
            pair_id, 
            data_array_new.shape[0], 
            data_array_new.shape[1], 
            data_array_new.tobytes(), 
            config, F, E, H, qvec, tvec
        ))

    # Insert the new data
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany("UPDATE keypoints SET rows=?, cols=?, data=? WHERE image_id=?", 
                          [(rows, cols, data, image_id) for image_id, rows, cols, data in keypoint_injection])
        cursor.executemany("UPDATE two_view_geometries SET rows=?, cols=?, data=? WHERE pair_id=?", 
                          [(rows, cols, data, pair_id) for pair_id, rows, cols, data, _, _, _, _, _, _ in two_view_geometry_injection])
        conn.commit()
    
    print("Successfully inserted manual matches into COLMAP database")



if __name__ == "__main__":
    db_path = "/tmp/colmap_tmp_webxu_gz/database.db"
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT image_id, rows, cols, data FROM keypoints")
        data_keypoints = cursor.fetchall()

        cursor = conn.cursor()
        cursor.execute("SELECT pair_id, rows, cols, data, config, F, E, H, qvec, tvec FROM two_view_geometries")
        data_two_view_geometries = cursor.fetchall()

    for image_id, rows, cols, data in data_keypoints:
        data_array = np.frombuffer(data, dtype=np.float32)
        data_array = data_array.reshape(rows, cols)
        print(f"keypoints for image {image_id}: {data_array.shape}")

    for pair_id, rows, cols, data, config, F, E, H, qvec, tvec in data_two_view_geometries:
        image_id_1, image_id_2 = pair_id_to_image_ids(pair_id)
        data_array = np.frombuffer(data, dtype=np.float32)
        data_array = data_array.reshape(rows, cols)
        print(f"two view geometry for {image_id_1} and {image_id_2}: {data_array.shape}")

