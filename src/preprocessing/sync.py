import cv2
import numpy as np
import tqdm
import os
import glob
from calibration_utils import get_camera_serial_number, load_calibration_data, get_iso, get_shutter, get_frame_rate
from sensor_data import timing_data

def find_rising_edge(brightness_values, dead_zone):
    # Find first value above dead_zone[1]
    idx_above = np.where(brightness_values > dead_zone[1])[0]
    if len(idx_above) == 0:
        return None
    idx_above = idx_above[0]
    # Find last value below dead_zone[0] that is before the first value above dead_zone[1]
    idx_below = np.where(brightness_values < dead_zone[0])[0]
    if len(idx_below) == 0:
        return None
    is_before = idx_below < idx_above
    idx_below = idx_below[is_before]
    if len(idx_below) == 0:
        return None
    idx_below = idx_below[-1]

    return idx_below

def gray_to_decimal(binary_gray):
    decimal = binary_gray
    while binary_gray > 0:
        binary_gray >>= 1
        decimal ^= binary_gray
    return decimal
    

def find_falling_edge(brightness_values, dead_zone):
    # Find last value above dead_zone[1]
    idx_above = np.where(brightness_values > dead_zone[1])[0]
    if len(idx_above) == 0:
        return None
    idx_above = idx_above[-1]
    # Find first value below dead_zone[0] that is after the last value above dead_zone[1]
    idx_below = np.where(brightness_values < dead_zone[0])[0]
    if len(idx_below) == 0:
        return None
    is_after = idx_below > idx_above
    idx_below = idx_below[is_after]
    if len(idx_below) == 0:
        return None
    idx_below = idx_below[0]
    return idx_below

if __name__ == "__main__":
    input_dir = "./data/input_video"
    input_id = 11
    
    # Find all camera IDs for the given input_id
    video_files = sorted(glob.glob(os.path.join(input_dir, f"{input_id:04d}_[0-9].mkv")))
    print(f"Found {len(video_files)} video files to process for input_id {input_id}")
    
    readout_time_per_line_us = 2.85
    tick_rate_us = 400
    dead_zone = (0.20, 0.5)
    period_us = 10 * tick_rate_us
    
    # Create output directory
    os.makedirs(os.path.join("data", "sync"), exist_ok=True)
    
    # Process each camera
    for input_path in video_files:
        # Extract camera_id from filename
        filename = os.path.basename(input_path)
        camera_id = int(filename.replace(".mkv", "").split("_")[1])
        
        print(f"\n{'='*60}")
        print(f"Processing: {filename} (input_id={input_id}, camera_id={camera_id})")
        print(f"{'='*60}")

        serial_number = get_camera_serial_number(input_path)
        calibration_data = load_calibration_data(serial_number)
        iso_exif = get_iso(input_path)
        shutter_exif = get_shutter(input_path)
        frame_rate_exif = get_frame_rate(input_path)

        frame_time_us = 1e6 / frame_rate_exif

        print(f"ISO exif: {iso_exif}, Shutter exif: {shutter_exif}, Frame rate exif: {frame_rate_exif}")
        timing_data_exif = timing_data[shutter_exif][iso_exif]
        readout_time_per_line_us = timing_data_exif["us_per_line"]
        exposure_us = timing_data_exif["exposure_us"]
        print(f"Readout time per line exif: {readout_time_per_line_us}, Exposure exif: {exposure_us}")

        brightness_data = np.load(os.path.join("data", "pattern_brightness", f"{input_id:04d}_{camera_id}.brightness.npz"))
        strip_brightness = brightness_data["strip_brightness"]
        marker_brightness = brightness_data["marker_brightness"]
        marker_points = brightness_data["marker_points"]
        strip_lines = brightness_data["strip_lines"][::-1]

        boot_pins = [0, 11]
        non_boot_pins = [pin for pin in range(marker_points.shape[0]) if pin not in boot_pins]
        marker_brightness_except_boot_pins = marker_brightness[:, non_boot_pins]

        first_frame = np.where(np.any(marker_brightness_except_boot_pins > 0.2, axis=1))[0][0]
        print(f"First frame: {first_frame}")

        pins_gray = list(range(marker_points.shape[0]-1))
        pin_transition_check = marker_points.shape[0]-1

        marker_points_gray = marker_points[pins_gray, :]
        marker_brightness_gray = marker_brightness[:, pins_gray]
        marker_points_transition_check = marker_points[pin_transition_check, :]
        marker_brightness_transition_check = marker_brightness[:, pin_transition_check]

        cap = cv2.VideoCapture(input_path)
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        num_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)

        exposures = []

        phases = np.zeros(num_frames_total)
        periods = np.zeros(num_frames_total)
        phases.fill(np.nan)
        periods.fill(np.nan)
        times_bookkeeping = np.zeros(num_frames_total)
        times_bookkeeping.fill(np.nan)

        num_false_frames = 0

        for i_frame in tqdm.tqdm(range(first_frame, num_frames_total), desc=f"Processing frames"):
            ret, frame = cap.read()
            if not ret:
                break
            
            strip_brightness_frame = strip_brightness[i_frame, :, :]
            lines_rising = []
            lines_falling = []
            times_rising = []
            times_falling = []
            for i in range(5):
                rising_edge = find_rising_edge(strip_brightness_frame[i, :], dead_zone)
                falling_edge = find_falling_edge(strip_brightness_frame[i, :], dead_zone)
                if rising_edge is not None:
                    y_coord = strip_lines[i, rising_edge, 1]
                    time = i * tick_rate_us * 2
                    lines_rising.append(y_coord)
                    times_rising.append(time)
                if falling_edge is not None:
                    y_coord = strip_lines[i, falling_edge, 1]
                    time = (i + 1) * tick_rate_us * 2
                    lines_falling.append(y_coord)
                    times_falling.append(time)
            times_rising = np.array(times_rising)
            times_falling = np.array(times_falling)
            lines_rising = np.array(lines_rising)
            lines_falling = np.array(lines_falling)

            # Find fully off segments
            fully_off_segments = []
            for i in range(5):
                is_fully_off = strip_brightness_frame[i, :] < dead_zone[0]
                y_coords = strip_lines[i, is_fully_off, 1]

                if np.any(is_fully_off):
                    y_min = np.min(y_coords)
                    y_max = np.max(y_coords)
                    segment_start = i * tick_rate_us * 2 - y_max * readout_time_per_line_us
                    segment_end = (i + 1) * tick_rate_us * 2 - y_min * readout_time_per_line_us
                    fully_off_segments.append((segment_start, segment_end))

            phase_us = None
            # No flanks
            distance_transform_resolution = 4096
            fully_off_discrete = np.ones(distance_transform_resolution, dtype=np.uint8) * 255
            
            def map_time_to_pixel(time):
                # 0 ... max_w-1
                # -period_us ... 2*period_us
                pixel = (time + period_us) * (distance_transform_resolution-1) / (3*period_us)
                return np.clip(int(pixel), 0, distance_transform_resolution-1)

            def map_pixel_to_time(pixel):
                time = pixel * 3*period_us / (distance_transform_resolution-1) - period_us
                return time

            for (segment_start, segment_end) in fully_off_segments:
                pixel_start = map_time_to_pixel(segment_start)
                pixel_end = map_time_to_pixel(segment_end)
                for period_offset in range(-1, 2):
                    pixel_start = map_time_to_pixel(segment_start + period_offset * period_us)
                    pixel_end = map_time_to_pixel(segment_end + period_offset * period_us)
                    fully_off_discrete[pixel_start:pixel_end] = 0
            
            distance_transform = cv2.distanceTransform(fully_off_discrete, cv2.DIST_L1, 0)

            min_index = map_time_to_pixel(0)
            max_index = map_time_to_pixel(period_us)
            max_distance_index = np.argmax(distance_transform[min_index:max_index]) + min_index
            interval_end = np.where(fully_off_discrete[max_distance_index:] == 0)[0][0]
            interval_start = np.where(fully_off_discrete[max_distance_index::-1] == 0)[0][0]
            interval_start = max_distance_index - interval_start + 1
            interval_end = max_distance_index + interval_end - 1
            time_start = map_pixel_to_time(interval_start)
            time_end = map_pixel_to_time(interval_end)
            
            max_distance_time = map_pixel_to_time(max_distance_index)

            if (time_end - time_start) > exposure_us*1.1 and i_frame > first_frame:
                phase_before = phases[i_frame-1]
                phase_predicted = (phase_before + frame_time_us) % period_us
                time_falling_projected_top = max_distance_time - exposure_us / 2
                if phase_predicted < time_falling_projected_top - period_us/2:
                    phase_predicted += period_us
                if phase_predicted > time_falling_projected_top + period_us/2:
                    phase_predicted -= period_us
                time_falling_projected_top = phase_predicted
            else:
                time_falling_projected_top = max_distance_time - exposure_us / 2
            
            phase_us = time_falling_projected_top % period_us

            # Grab gray code
            gray_code = marker_brightness_gray[i_frame, :] > 0.2
            gray_code_bits = gray_code[::-1].dot(1 << np.arange(gray_code.size)[::-1])
            is_transition = marker_brightness_transition_check[i_frame] > 0.2
            decimal_gray_code = gray_to_decimal(gray_code_bits)

            # Check if gray code is fully contained within a single period
            gray_code_times = marker_points_gray[:, 1] * readout_time_per_line_us + phase_us
            periods_start = gray_code_times // period_us
            periods_end = (gray_code_times + exposure_us) // period_us

            is_contained = np.all(periods_start == periods_end)
            if is_contained:
                if (phase_us // period_us) < periods_start[0]:
                    decimal_gray_code -= 1
            elif (is_transition != (decimal_gray_code % 2 == 0)):
                decimal_gray_code -= 1

            phases[i_frame] = phase_us
            periods[i_frame] = decimal_gray_code
            times_bookkeeping[i_frame] = phase_us + decimal_gray_code * period_us

            if i_frame > first_frame:
                time_predicted = times_bookkeeping[i_frame-1] + frame_time_us
                time_read = times_bookkeeping[i_frame]
                time_diff = np.abs(time_predicted - time_read)
                time_diff_relative = time_diff / frame_time_us
                if time_diff_relative > 0.1:
                    print(f"Time deviated from prediction by {time_diff*1e-3:.2f} ms")
                    times_bookkeeping[i_frame] = time_predicted
                    num_false_frames += 1

        print(f"Number of false frames: {num_false_frames}")

        output_file = os.path.join("data", "sync", f"{input_id:04d}_{camera_id}.times.npz")
        np.savez(output_file, times=times_bookkeeping, exposure_us=exposure_us, readout_time_per_line_us=readout_time_per_line_us, period_us=period_us)
        print(f"Saved to {output_file}")

        print(f"Time range: {(np.nanmax(times_bookkeeping) - np.nanmin(times_bookkeeping))*1e-6:.2f} s")
        
        cap.release()
    
    print(f"\n{'='*60}")
    print(f"Finished processing all {len(video_files)} camera files")
    print(f"{'='*60}")