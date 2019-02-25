import glob
import os
import cv2
import numpy as np
import h5py
import copy

"""

"""


class VideoProcessor:
    def __init__(self, num_frames_before_after, enhance_clahe=False):
        # print "\nStarting processing videos and extracting frames and data..."
        self.frame_data = []
        #self.scale_factor = np.float(1) / self.args.down_sample
        self.num_frames_before_after = num_frames_before_after
        is_include_mid_in_frames = True
        if is_include_mid_in_frames: self.num_frames_before_after[1] += 1
        #if args.is_split:
        #    self.h5_size = args.h5_size
        self.image_shape = (480, 640) # TODO:
        self.enhance_clahe = enhance_clahe

    def enhance(self, image):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe_1 = clahe.apply(image[:, :, 0])
        img_clahe_2 = clahe.apply(image[:, :, 1])
        img_clahe_3 = clahe.apply(image[:, :, 2])
        img_clahe = cv2.merge((img_clahe_1, img_clahe_2, img_clahe_3))
        return img_clahe

    def preprocess_image(self, image):
        # if self.enhance_clahe:
        #     image = self.enhance(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (self.image_shape[1], self.image_shape[0]), interpolation=cv2.INTER_NEAREST)
        return image

    def get_video_frames(self, cap, frame_number):
        frame_names = []
        frame_images = np.zeros((self.num_frames_before_after[1]*2 - 1, self.image_shape[0], self.image_shape[1]), dtype=np.uint8)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        success = frame_number + self.num_frames_before_after[0] > 0 and n_frames - frame_number > self.num_frames_before_after[1]
        if success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number + self.num_frames_before_after[0])
            for i in range(self.num_frames_before_after[0], self.num_frames_before_after[1]): #+ 1):
                ret, frame = cap.read()
                if not ret: 
                    ret, frame = cap.read()
                    if not ret:
                        return None, None
                        # raise Exception("VIDEO MIGHT BE CORRUPT!")
                frame = self.preprocess_image(frame)
                frame_name = str(frame_number + i) + '.png'
                frame_names.append(frame_name)
                #frame_images = np.concatenate((frame_images, frame[np.newaxis, :]), axis=0)
                frame_images[i + self.num_frames_before_after[0], ...] = frame

            #print("   > {0} frames added.".format(frame_images.shape[0]))
            return frame_images, frame_names
        else:
            print("Warning: image is skipped since frame number is {0}".format(frame_number))
            return None, None


    def read_all_samples_in_video(self, folder_path, video_path):
        """
        folder_path includes both xml and labelled images.
        video_path to a .avi file
        """
        # Optimize: instead of concatenating, allocate frame_images straight away

        new_folder = os.path.abspath(os.path.join(folder_path, os.pardir))
        new_folder = new_folder.replace("Dataset", "DatasetFlownet")
        new_folder = os.path.join(new_folder, "FlownetImage")
        if not os.path.exists(new_folder):
            print("Creating a new folder at: ", new_folder)
            os.makedirs(new_folder)
        #else:
        #    raise Exception("Folder already exists!")

        number_of_frames = np.sum([abs(number) for number in self.num_frames_before_after])

        image_data = np.zeros((0, 1, self.image_shape[0], self.image_shape[1]), dtype=np.uint8)
        #frame_data = np.zeros((0, number_of_frames, self.image_shape[0], self.image_shape[1]), dtype=np.uint8)
        frame_data_names = []
        img_path = glob.glob(os.path.join(*[folder_path, '*.png']))
        frame_data = np.zeros((len(img_path), number_of_frames, self.image_shape[0], self.image_shape[1]), dtype=np.uint8)
        inds_to_be_deleted = []
        if os.path.isfile(video_path):
            cap = cv2.VideoCapture(video_path)
            for i in range(len(img_path)):
                try:
                    frame_number = int(img_path[i].split("_")[-1].split(".")[0])
                except:
                    print("Image is wrong! Doesn't contain a frame number!")
                    continue
                img = cv2.imread(img_path[i])
                img = self.preprocess_image(img)
                image_data = np.concatenate((image_data, img[np.newaxis, np.newaxis, ...]), axis=0)

                img_name = os.path.basename(os.path.normpath(img_path[i]))
                full_image_name = img_name.split("_")[:-1]
                full_image_name = '_'.join(full_image_name)
                #frame_images = np.zeros((0, self.image_shape[0], self.image_shape[1]), dtype=np.uint8)
                frame_names = []
                frame_images, frame_names = self.get_video_frames(cap, frame_number, full_image_name, new_folder, frame_names, is_save_png_images=False)
                if frame_images is None:
                    inds_to_be_deleted.append(i)
                    continue
                else:
                    #frame_data = np.concatenate((frame_data, frame_images[np.newaxis, :]), axis=0)
                    frame_data[i, ...] = frame_images
                    frame_data_names.append(copy.deepcopy(frame_names))
            if inds_to_be_deleted != []:
                frame_data = np.delete(frame_data, inds_to_be_deleted, axis=0)
            cap.release()
            frame_data_names = np.array(frame_data_names)
            file_path = os.path.join(new_folder, full_image_name + ".h5")
            comp_kwargs = {'compression': 'gzip', 'compression_opts': 2}
            with h5py.File(file_path, 'w') as f: # TODO: Add compression
                f.create_dataset('data', data=image_data, **comp_kwargs)
                f.create_dataset('frames', data=frame_data, **comp_kwargs)
                f.create_dataset('frame_names', data=frame_data_names, **comp_kwargs)
            print("\nFolder {0}\n with {1} images and {2} frames each is done!".format(folder_path, len(img_path), number_of_frames))

        return image_data, frame_data, frame_data_names


    def process_all_videos(self, xml_paths, video_paths):
        for xml_path, video_path in zip(xml_paths['train_val'], video_paths['train_val']):
            self.read_all_samples_in_video(xml_path, video_path)

        for xml_path, video_path in zip(xml_paths['test'], video_paths['test']):
            self.read_all_samples_in_video(xml_path, video_path)

        return 






    def split_data_concatenation(self, label_fld, video_fld, k, folder_type):
        image_data, label_data, mask_data = [], [], []
        number_of_frames = np.sum([abs(number) for number in self.num_frames_before_after])
        if self.network_type == "sk":
            image_data, label_data, mask_data = Network(self.args).sk_data_initialization
        elif self.network_type == "unet":
            image_data, label_data, mask_data, _, _, _ = Network(self.args).unet_data_initialization
        frame_data = np.zeros((0, number_of_frames, image_data.shape[2], image_data.shape[3]), dtype='float32')
        for l in range(len(label_fld)):
            label_name = label_fld[l]
            frame_number = int(label_name.split("_")[-1].split(".")[0])
            video_file = video_fld[l]
            if os.path.isfile(video_file):
                cap = cv2.VideoCapture(video_file)
                frame_images = np.zeros((0, image_data.shape[2], image_data.shape[3]), dtype='float32')
                frame_images = self.get_video_frames(cap, frame_number, frame_images)
                if frame_images is None:
                    continue
                elif self.object_to_detect == "bucketpattern":
                    image_data, label_data, mask_data = BucketPattern(self.args).read_bucketpattern_image_label_mask(label_name, image_data, label_data, mask_data)
                elif self.object_to_detect == "teethline" or self.object_to_detect == "teeth":
                    image_data, label_data, mask_data = TeethlineTeeth(self.args).read_teeth_teethline_image_label_mask(label_name, image_data, label_data, mask_data)
                frame_data = np.concatenate((frame_data, frame_images[np.newaxis, :]), axis=0)
                cap.release()
        print(" * The {0}th split with {1} images is done".format(k + 1, len(label_fld)))
        if folder_type:
            _, n_train, n_test, total_samples = H5Maker(self.args).split_data_to_train_validation(image_data)
            H5Maker(self.args).save_h5(image_data, label_data, mask_data, n_train, n_test, total_samples, '_' + str(k + 1), folder_type, frame_data)
        else:
            H5Maker(self.args).save_h5(image_data, label_data, mask_data, 0, image_data.shape[0], image_data.shape[0], '_' + str(k + 1), folder_type, frame_data)
        del image_data, label_data, mask_data, frame_data

    def make_unit_h5(self, folders, videos, folder_type=1):
        pool = Pool(processes=mp.cpu_count()-1)
        result_list = []
        for img_fld in range(len(folders)):
            #result_list.append(self.unit_data_concatenation(folders[img_fld], videos[img_fld]))
            pool.apply_async(self.unit_data_concatenation, args=(folders[img_fld], videos[img_fld],), callback=result_list.append)
        pool.close()
        pool.join()
        output_e = [t for t in result_list if t != ()]
        if len(output_e) == 0:
            name = "Train-Validation" if folder_type else "Test"
            print("Error: there is no image folder for {0}".format(name))
            return
        print("number of non empty folders processed:{0}".format(len(output_e)))
        self.image_data = np.vstack([item[0] for item in output_e])
        self.label_data = np.vstack([item[1] for item in output_e])
        self.mask_data = np.vstack([item[2] for item in output_e])
        self.frame_data = np.vstack([item[3] for item in output_e])
        if folder_type:
            rand_perm_data, n_train, n_test, total_samples = H5Maker(self.args).split_data_to_train_validation(self.image_data)
            self.image_data = self.image_data[rand_perm_data, :, :, :]
            self.label_data = self.label_data[rand_perm_data,]
            self.mask_data = self.mask_data[rand_perm_data,]
            self.frame_data = self.frame_data[rand_perm_data,]
            H5Maker(self.args).save_h5(self.image_data, self.label_data, self.mask_data, n_train, n_test, total_samples, "", folder_type, self.frame_data)
        else:
            H5Maker(self.args).save_h5(self.image_data, self.label_data, self.mask_data, 0, self.image_data.shape[0], self.image_data.shape[0], "", folder_type, self.frame_data)
        del self.image_data, self.mask_data, self.label_data, self.frame_data

    def make_split_h5(self, folders, videos, folder_type=1):
        ind_mat, video_int_mat = SplitH5(self.args).create_index(folders, videos)
        SplitH5(self.args).randomize_mat(ind_mat)
        SplitH5(self.args).randomize_mat(video_int_mat)
        print("splitting images ... ")
        sub_folders = []
        sub_folders_videos = []
        k = 0
        while k * self.h5_size < len(ind_mat):
            sub_folders.append(SplitH5(self.args).split_image_paths(ind_mat, k))
            sub_folders_videos.append(SplitH5(self.args).split_image_paths(video_int_mat, k))
            k += 1
        pool = Pool(processes=mp.cpu_count()-1)
        k = 0
        for fld in range(len(sub_folders)):
            img_fld = sub_folders[fld]
            video_fld = sub_folders_videos[fld]
            pool.apply_async(self.split_data_concatenation, args=(img_fld, video_fld, k, folder_type,))
            # self.split_data_concatenation(img_fld, video_fld, k, folder_type)
            k += 1
        pool.close()
        pool.join()
