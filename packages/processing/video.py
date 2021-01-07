from scipy.fftpack import idct
import cv2
import numpy as np

def preprocess_ntcd_matlab(video_writer, matlab_frames, frame, width, height, y_hat_hard=None):
    data_frame = matlab_frames[frame]  # data frame will be shortened to "df" below
    reshaped_df = data_frame.reshape(width, height)
    idct_df = idct(idct(reshaped_df).T).T
    # normalized_df = idct_df / (matlab_frames_list_per_user.flatten().max() + epsilon) * 255.0
    normalized_df = cv2.normalize(idct_df, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    rotated_df = np.rot90(normalized_df, 3)

    # Add label on the video
    if y_hat_hard is not None and y_hat_hard[frame] == 1:
        rotated_df[-9:,-9:] = 9*[255] # white square
    
    rgb_rotated_df = cv2.merge([rotated_df] * 3)
    video_writer.write(rgb_rotated_df)

def preprocess_ntcd_matlab_nowriter(matlab_frames, frame, width, height, y_hat_hard=None):
    data_frame = matlab_frames[frame]  # data frame will be shortened to "df" below
    reshaped_df = data_frame.reshape(width, height)
    idct_df = idct(idct(reshaped_df).T).T
    # normalized_df = idct_df / (matlab_frames_list_per_user.flatten().max() + epsilon) * 255.0
    normalized_df = cv2.normalize(idct_df, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    rotated_df = np.rot90(normalized_df, 3)

    # Add label on the video
    if y_hat_hard is not None and y_hat_hard[frame] == 1:
        rotated_df[-9:,-9:] = 9*[255] # white square
    
    rgb_rotated_df = cv2.merge([rotated_df] * 3)
    return rgb_rotated_df