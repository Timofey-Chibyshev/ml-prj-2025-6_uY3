import matplotlib.pyplot as plt
import random

def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(10,7))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 6);        
        index += 1
    plt.show()

def visualize(x_train, y_train, x_test, y_test):
    images_2_show = []
    titles_2_show = []
    for i in range(0, 10):
        r = random.randint(1, 60000)
        images_2_show.append(x_train[r])
        titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))    

    for i in range(0, 5):
        r = random.randint(1, 10000)
        images_2_show.append(x_test[r])        
        titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))    

    show_images(images_2_show, titles_2_show)

if __name__ == "__main__":
    from data_loader import loader

    (x_train, y_train), (x_test, y_test) = loader.load_data()
    visualize(x_train, y_train, x_test, y_test)
