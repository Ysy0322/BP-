import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class BmpReader {


    public static int[] read(String filepath) {
        File file = new File(filepath);
        BufferedImage bi = null;
        try {
            bi = ImageIO.read(file);
        } catch (Exception e) {
            e.printStackTrace();
        }

        if (bi == null) {
            System.out.println("Error");
        }

        int width = 28;
        int height = 28;

        int[][] img = new int[height][width];

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int pixel = bi.getRGB(i, j);
                if (pixel == -1) {
                    img[j][i] = 1;
                } else {
                    img[j][i] = 0;
                }
            }
        }
        //转化成一维数组
        return transform(img);
    }

    public static int[] transform(int[][] img) {
        int[] result = new int[784];
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                result[i * 28 + j] = img[i][j];
            }
        }
        return result;
    }

}
