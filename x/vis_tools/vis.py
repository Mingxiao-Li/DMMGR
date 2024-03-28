import cv2
import matplotlib.pyplot as plt


def draw_region_in_img(
    img,
    locations,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    ret_color=(0, 0, 255),  # red
    ret_width=2,
    txt_color=(0, 0, 255),
    txt_size=1,
    txt_width=2,
    save_path=None,
    pred_prob=None,
    region_tag=None,
):

    img = cv2.imread(img)
    for i in range(len(locations)):
        x1, y1, x2, y2 = locations[i]
        a = (int(x1), int(y1))
        b = (int(x2), int(y2))
        cv2.rectangle(img, a, b, ret_color, ret_width)
        if region_tag is not None:
            obj = region_tag[i]
            if pred_prob is not None:
                show_text = "{} {:.3f}".format(obj, pred_prob[i])
            else:
                show_text = obj
            img = cv2.putText(img, show_text, a, font, txt_size, txt_color, txt_width)
    if save_path is not None:
        cv2.imread(save_path, img)
    else:
        img = img[:, :, ::-1]
        plt.imshow(img)
        plt.show()