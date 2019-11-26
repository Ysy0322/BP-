import java.io.File;

public class ReadImages {
     int[][] train_data;
     int[] train_label;
     int[][] test_data;
     int[] test_label;

    public int[] getTest_label() {
        return test_label;
    }

    public int[][] getTest_data() {
        return test_data;
    }

    public int[] getTrain_label() {
        return train_label;
    }

    public int[][] getTrain_data() {
        return train_data;
    }

    public String[] get_image_paths(String path) {
        File file = new File(path);
        File[] arr = file.listFiles();
        String[] image_paths = new String[arr.length];
        for (int i = 0; i < image_paths.length; i++) {
            image_paths[i] = arr[i].getPath();
        }
        return image_paths;
    }

    public int[][][] get_total_images() {

        String[][] images_paths = new String[14][];
        int[][][] image_info = new int[14][][];
        for (int i = 0; i < 14; i++) {
            images_paths[i] = get_image_paths("TRAIN\\" + (i + 1));
            image_info[i] = new int[images_paths[i].length][];
            for (int j = 0; j < images_paths[i].length; j++) {
                int[] image = BmpReader.read(images_paths[i][j]);
                image_info[i][j] = new int[image.length];
                for(int k=0;k<image.length;k++){
                    image_info[i][j][k] = image[k];
                }

            }
        }
        return image_info;
    }

    public void set_data_label(double percentage) {

        int[][][] image_info = get_total_images();
        int length = (int) (image_info[0].length * percentage);
        this.train_data = new int[length * 14][784];
        this.train_label = new int[length * 14];

        int index = 0;
        for (int i = 0; i < length; i++) {
            for (int j = 0; j < 14; j++) {
                index = 14 * i + j;
                train_data[index] = image_info[j][i];
                train_label[index] = j;
            }
        }
        int remain_length = image_info[0].length - length;
        this.test_data = new int[remain_length * 14][784];
        this.test_label = new int[remain_length * 14];
        for (int i = 0; i < remain_length; i++) {
            for (int j = 0; j < 14; j++) {
                index = 14 * i + j;
                int img_index = length+i;
                test_data[index] = image_info[j][img_index];
                test_label[index] = j;
            }
        }
    }

}
