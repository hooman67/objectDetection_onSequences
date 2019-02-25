import os
from time import time

import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def convert_to_pixels(bbox, image_w=640, image_h=640):
    xmin = bbox.xmin * image_w
    xmax = bbox.xmax * image_w
    ymin = bbox.ymin * image_h
    ymax = bbox.ymax * image_h
    return int(round(xmin)), int(round(xmax)), int(round(ymin)), int(round(ymax))


def test_frame_selector():
    from utils import BoundBox
    skip_rate = 3
    frame_rate = 30
    Ns, Ks, Ts = 70, 30, 35
    print("Ns = %d, Ks = %d, Ts = %d" % (Ns, Ks, Ts))
    N, K = Ns * frame_rate / skip_rate, Ks * frame_rate 
    T = Ts * frame_rate / skip_rate
    print("N = %d, K = %d, T = %d" % (N, K, T))
    non_None_threshold = 1
    fm_frame_selector = FmFrameSelector(N, K, T, "",
				      label_ind=3,
				      non_None_threshold=non_None_threshold)
    i = 0
    image = np.zeros((640, 640, 3))
    num_teeth = 6
    corr_bbox1 = BoundBox(0.1, 0.1, 0.2, 0.2, 0.6, [0., 0.08, 0., 0.92, 0.], label=3)
    bboxes = [BoundBox(0.1, 0.1, 0.2, 0.2, 0.9, 2, label=2),
              BoundBox(0.1, 0.1, 0.2, 0.2, 0.9, 2),
              BoundBox(0.1, 0.1, 0.2, 0.2, 0.9, 1),
              corr_bbox1,
              BoundBox(0.2, 0.2, 0.4, 0.4, 0.5, 4, label=0)]
    fm_frame_selector.update(i, image, bboxes, num_teeth,
                             write_selection=False)
    assert fm_frame_selector.bbox_buffer[0] == corr_bbox1 

    i += 1
    bboxes = []
    fm_frame_selector.update(i, image, bboxes, num_teeth,
                             write_selection=False)

    i += 1
    corr_bbox2 = BoundBox(0.1, 0.1, 0.2, 0.2, 0.22, [0.05, 0., 0., 0.90, 0.05], label=3)
    bboxes = [BoundBox(0.1, 0.1, 0.2, 0.2, 0.9, 2, label=2),
              BoundBox(0.1, 0.1, 0.2, 0.2, 0.9, 2),
              BoundBox(0.1, 0.1, 0.2, 0.2, 0.9, 1),
              corr_bbox2,
              BoundBox(0.2, 0.2, 0.4, 0.4, 0.5, 4, label=0)]
    fm_frame_selector.update(i, image, bboxes, num_teeth,
                             write_selection=False)
    assert fm_frame_selector.bbox_buffer[-1] == corr_bbox2
    assert len(fm_frame_selector.frame_buffer) == 3
    assert len(fm_frame_selector.bbox_buffer) -\
            fm_frame_selector.bbox_buffer.count(None) == 2

    frame, bbox, ind = fm_frame_selector.select_frame()
    assert bbox == corr_bbox1
    assert ind == 0

    for j in range(100):
        i += 1
        bboxes = []
        fm_frame_selector.update(i, image, bboxes, num_teeth,
                                 write_selection=False)
    frame, bbox, ind = fm_frame_selector.select_frame()
    assert bbox == corr_bbox1
    assert ind == 0


class FmFrameSelector:

    def __init__(
        self,
        maxLength_frameBuffer,
        frequeny_decisionMaking,
        timeWindow_selectFrame,
        save_folder,
        label_ind,
        stride,
        minNumber_validScoresToSlectFrame=15):
        

        #index of interesting lebel as defined in config
        self.label_ind = label_ind
        self.minNumber_validScoresToSlectFrame = minNumber_validScoresToSlectFrame
        self.stride = stride

        self.window_ind = 0  # index of window selection

        # setup skip-rate stuff
        self.maxLength_frameBuffer = maxLength_frameBuffer  # max length of frame_buffer and bbox_buffer
        self.frequeny_decisionMaking = frequeny_decisionMaking  # every K-th frame a decision is made
        self.timeWindow_selectFrame = timeWindow_selectFrame  # time window that's taken in select_frame()
        
        
        #set up the buffers
        self.bbox_buffer = []  # this list must always be <=N long
        self.frame_buffer = []  # list of numpy frames, must be == len(bbox_buffer)
        self.flow_buffer = []  # list of dense opencv flows, must be len(box_buffer)-1

        self.frames_selected = []
        self.selected_frame_numbers = []

        #These are for MatInside
        self.min_aspect_ratio = 0.8
        self.max_aspect_ratio = 2.6


        #setup the folder to save results at
        label_folder_name = "FM"
        save_folder = os.path.join(save_folder, label_folder_name)
        self.save_folder = save_folder
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        

    def add_bbox(self, bboxes):
        #adds the bbox for the selected label (either matInside or WmInside) the bbox buffer
        bbox_to_add = None
        for bbox in bboxes:
            if bbox.get_label() == self.label_ind:
                bbox_to_add = bbox

        self.bbox_buffer.append(bbox_to_add)

        #if the length of the buffer is too long. Remove the first element from it.
        if len(self.bbox_buffer) == self.maxLength_frameBuffer+1:    
            self.bbox_buffer = self.bbox_buffer[1:]

    
    def add_frame(self, frame):
        #adds this to the frame buffer
        self.frame_buffer.append(frame)
        
        #if the length of the buffer is too long. Remove the first element from it.
        if len(self.frame_buffer) == self.maxLength_frameBuffer+1:
            self.frame_buffer = self.frame_buffer[1:]


    def is_shovel_idle(self, frame_slice, is_plot=False):
        """ Compares flow of first and last frame in a slice. """
        first_frame, last_frame = frame_slice[0], frame_slice[-1] 
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(first_frame, last_frame, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        median_mag = np.median(np.abs(mag), axis=(0, 1))
        mean_mag = np.mean(np.abs(mag), axis=(0, 1))

        if is_plot:
            hsv = np.zeros_like(first_frame)
            # mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            # hsv[...,0] = ang*180/np.pi/2  # ignore angle
            hsv = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_GRAY2RGB)
            fig, ax = plt.subplots(1, 3, figsize=(14, 8))
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_GRAY2RGB)
            last_frame = cv2.cvtColor(last_frame, cv2.COLOR_GRAY2RGB)
            ax[0].imshow(first_frame)
            ax[1].imshow(last_frame)
            ax[2].imshow(rgb)
            fig.suptitle("Median flow magnitude %.2f. Mean %.2f" %\
                         (median_mag, mean_mag))
            plt.show()

        #mean magnitude of the flow
        pixel_threshold = 1
        if mean_mag >= pixel_threshold:
            return False
        else:
            return True


    def argsort(self, seq):
        """ Python version of numpy's argsort. 
        Returns a list of indices in ascending order. None is assumed to be
        the lowest number. """
        
        #print("*******HS argsort*******")
        #print(seq)
        
        #return sorted(range(len(seq)), key=seq.__getitem__)
        #return [x for x,y in sorted(enumerate(seq), key = lambda x: x[1])]
        #return sorted(range(len(seq)), key = lambda x: seq[x].sort_property)
        seq_npflt = np.array(seq).astype(np.float)
        nanCount = np.count_nonzero(np.isnan(seq_npflt))

        sortedInd = np.argsort(seq_npflt)
        sortedInd_rotatedByNan = np.roll(sortedInd, nanCount)

        return sortedInd_rotatedByNan.tolist(), nanCount


    def is_legitimate_bbox(self, bbox):
        """ Checks several constraints (e.g. xmax > xmin, etc...)  """
        if bbox is None: return None
        aspect_ratio = float(bbox.xmax - bbox.xmin) / (bbox.ymax - bbox.ymin)
        is_legit = bbox.xmax > bbox.xmin and bbox.ymax > bbox.ymin and\
                   aspect_ratio > self.min_aspect_ratio and\
                   aspect_ratio < self.max_aspect_ratio
        return is_legit


    def save_img(self, folder_to_save, img, j, score=None):
        folder_path = os.path.join(self.save_folder,  folder_to_save)
        create_folder(folder_path)
        name = str(self.window_ind) + "_" + str(j) + ".jpg"
        filepath = os.path.join(folder_path, name)
        if score:
            cv2.putText(img, 
                        str(int(round(score, -2))),
                        (590, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.4e-3 * 640, 
                        (0, 255, 0), 2)
        ret = cv2.imwrite(filepath, img)
        if ret is None:
            raise
        

    def time_suppress(self, is_plot=True):
        """ Supresses the bboxes which are single detections in the time domain.
        e.g. a single detection while in a neighbouring frame on both sides 
        there isn't a detection."""
        time_suppressed = []
        for i in range(2, len(self.bbox_buffer)-2):
            curr_bbox = self.bbox_buffer[i]
            is_all_neighbours_None = True
            for j in [-1, 1]:
                bbox_to_compare = self.bbox_buffer[i+j]
                if bbox_to_compare is not None: is_all_neighbours_None = False
            if is_all_neighbours_None:
                if is_plot and self.bbox_buffer[i] is not None:
                    self.save_img("time_suppressed", self.frame_buffer[i], i)
                self.bbox_buffer[i] = None


    def compute_selection_scores(self, bboxes, is_plot=True):
        selection_scores = []
        image_height = 640
        for i, bbox in zip(range(len(bboxes)), bboxes):
            if bbox is None or not self.is_legitimate_bbox(bbox):
                selection_score = None
                selection_scores.append(selection_score)
                continue
            xmin, xmax, ymin, ymax = convert_to_pixels(bbox,
                                                       image_w=640, image_h=640)
            area = (xmax - xmin) * (ymax - ymin) 
            score = bbox.get_score()
            y_center = ymax - (float(ymax - ymin)/2)
            selection_score = (score**2) * np.sqrt(area) *\
                               max((image_height - y_center, 0.))
            weights = [1., 1., 1.]
            selection_score2 = weights[0] * score + weights[1] * np.sqrt(area) +\
                              weights[2] * max((image_height - y_center, 0.))
            # print("%.3f, %.3f, %.1f, %.1f, %.1f, %.1f " %\
            #         (score, score**2, np.sqrt(area), max((640 - y_center, 0.)),
            #         selection_score, selection_score2))
            selection_scores.append(selection_score)
            if is_plot:
                self.save_img("all_scores", self.frame_buffer[-self.timeWindow_selectFrame:][i], i, selection_score)

        return selection_scores


    def select_frame(self):
        """ Returns one selected frame and corresponding bbox. """
        self.time_suppress()
        
        #select a decision making window
        bbox_slice = self.bbox_buffer[-self.timeWindow_selectFrame:]  # returns a slice of size of window
        frame_slice = self.frame_buffer[-self.timeWindow_selectFrame:]
        

        if self.is_shovel_idle(frame_slice):
            return None, None, None, None


        # main logic of frame selection
        else:  
            selection_scores = self.compute_selection_scores(bbox_slice)

            sorted_inds, numberOfNoneScores = self.argsort(selection_scores)


            numberOfValidScores = len(sorted_inds) - numberOfNoneScores
            
            if numberOfValidScores < self.minNumber_validScoresToSlectFrame:
                return None, None, None, None
            


            #Pick the highest Score
            ind_selected_max = sorted_inds[-1]
            
            #Pick the 90thPercentile Score
            ninetieth_percentile_ind = int(round(numberOfValidScores * 0.9)) + numberOfNoneScores - 1
            ind_selected = sorted_inds[ninetieth_percentile_ind]



            bbox_selected = bbox_slice[ind_selected]
            frame_selected = frame_slice[ind_selected]
            self.frames_selected.append(frame_selected)
            
            frame_selected_max = frame_slice[ind_selected_max]

            return frame_selected, bbox_selected, ind_selected, ind_selected_max, frame_selected_max
            

    def update(self, i, frame, list_of_bboxes, num_bucket_teeth, write_selection=True):
        self.add_frame(frame)
        self.add_bbox(list_of_bboxes)

        assert(len(self.frame_buffer) == len(self.bbox_buffer))

        
        #If it's time to make a decision
        if i % self.frequeny_decisionMaking == 0 and i != 0:
            print('trying to select frames')
            
            selected_frame, selected_bbox, ind_selected, frame_max, ind_delected_max = self.select_frame()
            
            if selected_frame is None:
                print("No frames selected at point %d!" % i)
                pass

            else:
                if len(self.frame_buffer) < self.maxLength_frameBuffer:
                    frame_num = ind_selected*self.stride + (len(self.frame_buffer) - self.timeWindow_selectFrame)*self.stride
                else:
                    frame_num = i - (self.timeWindow_selectFrame*self.stride) + (ind_selected)*self.stride

                self.selected_frame_numbers.append(frame_num)


            if selected_frame is not None and write_selection:
                self.selected_frame_numbers.append(frame_num)
                # print("Frame Selected at %d at point %d."  % (frame_num, i))
                filepath = os.path.join(self.save_folder,
                    str(self.window_ind) + "_" + str(i) + "_" +\
                            str(frame_num) + ".jpg")
                cv2.imwrite(filepath, selected_frame) 
                filepath_max = os.path.join(self.save_folder,
                    str(self.window_ind) + "_" + str(i) + "_" +\
                            str(frame_num) + "_max" + ".jpg")
                cv2.imwrite(filepath_max, frame_max) 
            self.window_ind += 1


class RandomFmFrameSelector(FmFrameSelector):
    """ Randomly selects a frame with a bbox of class label_ind available """

    def __init__(
        self,
        maxLength_frameBuffer,
        frequeny_decisionMaking,
        timeWindow_selectFrame,
        save_folder,
        label_ind,
        stride,
        minNumber_validScoresToSlectFrame=15,
        selection_option="random"):
        

        FmFrameSelector.__init__(
            self,
            maxLength_frameBuffer,
            frequeny_decisionMaking,
            timeWindow_selectFrame,
            save_folder,
            label_ind,
            stride,
            minNumber_validScoresToSlectFrame=15)

        self.save_folder += "_" + selection_option
        self.selection_option = selection_option
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)


    def select_frame(self):
        if self.frame_buffer == []:
            return None, None
        bbox_slice = self.bbox_buffer[-self.timeWindow_selectFrame:]
        frame_slice = self.frame_buffer[-self.timeWindow_selectFrame:]
        num_None = bbox_slice.count(None)
        num_non_None = len(bbox_slice) - num_None
        if num_non_None <= self.minNumber_validScoresToSlectFrame:
            return None, None
        if self.selection_option == "random":
            uniform_prob = 1. / num_non_None
            probs = [0. if el is None else uniform_prob for el in bbox_slice]
        elif self.selection_option == "weighted":
            probs = [0. if el is None else el.get_score() for el in bbox_slice]
            normalizing_factor = 1. / sum(probs)
            probs = [normalizing_factor * p for p in probs]
        else:
            raise ValueError("Wrong argument selection_option!")
        rand_ind = np.random.choice(range(len(bbox_slice)), size=1, p=probs)[0]
        rand_bbox = bbox_slice[rand_ind]
        rand_frame = frame_slice[rand_ind]
        return rand_frame, rand_ind


    def update(self, i, frame, list_of_bboxes, write_selection=True):
        self.add_frame(frame)
        self.add_bbox(list_of_bboxes)
        if i % self.frequeny_decisionMaking == 0 and i != 0:
            selected_frame, ind_selected = self.select_frame()
            if selected_frame is None:
                # print("No selection at point %d!" % i)
                pass
            else:
                if len(self.frame_buffer) < self.maxLength_frameBuffer:
                    frame_num = ind_selected*3
                else:
                    frame_num = i - (self.timeWindow_selectFrame*3) + (ind_selected)*3
                self.selected_frame_numbers.append(frame_num)

            if selected_frame is not None and write_selection:
                if len(self.frame_buffer) < self.maxLength_frameBuffer:
                    frame_num = ind_selected*3
                else:
                    frame_num = i - (self.timeWindow_selectFrame*3) + (ind_selected)*3
                # print(i, self.timeWindow_selectFrame, ind_selected, self.timeWindow_selectFrame*3, ind_selected*3)
                filepath = os.path.join(self.save_folder,
                                        str(i) + "_" + str(frame_num) + ".jpg")

                # print("%s frame Selected at frame %d at point %d."  %\
                #         (self.selection_option, frame_num, i))
                # print("Random frame Selected at frame %d and path %s."  %\
                #         (frame_num, filepath))
                cv2.imwrite(filepath, selected_frame) 


