import matplotlib.pyplot as plt
import random

portion = 4

def show_images(images, title_texts):
    rows = len(images)
    plt.figure(figsize=(12,9))
    plt.subplots_adjust(hspace=2.0)
    index = 1    
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, 1, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 12);        
        index += 1
    plt.tight_layout()
    plt.show()

def visualize(x_train, y_train, x_test, y_test):
    images_2_show = []
    titles_2_show = []
    for i in range(0, portion):
        r = random.randint(0, len(y_train) - 1)
        images_2_show.append(x_train[r])
        titles_2_show.append(y_train[r]) 

    for i in range(0, portion):
        r = random.randint(0, len(y_test) - 1)
        images_2_show.append(x_test[r])        
        titles_2_show.append(y_test[r])    

    show_images(images_2_show, titles_2_show)

if __name__ == "__main__":
    from data_loader import loader

    (x_train, y_train), (x_test, y_test) = loader.load_data()
    visualize(x_train, y_train, x_test, y_test)
