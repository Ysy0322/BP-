
import java.util.Random;

public class BPNetwork {
    int input_n = 0;
    int[] input_cells;
    int output_n = 0;
    double[] output_cells;
    double[][] input_w;
    double[][]output_w;
    int[] hidden_ns;
    double[][][]hidden_ws;
    double[][] hidden_bs;
    double[] output_b;
    double[][] hidden_results;
    double[] output_deltas;
    double[][] hidden_deltases;
    double[] predict;
    int[][] test_datas;
    int[] test_label;
    int[][] train_datas;
    int[] train_label;
     Random random = new Random(0);
    private double rand(double a,double b){
        return (b-a) * random.nextDouble()+a;
    }
    private double[][] generate_w(int m,int n){
        double[][] w = new double[m][n];
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                w[i][j] = rand(-0.9,0.9);
            }
        }
      return w;
    }
    private double[] generate_b(int m){
        double[] b = new double[m];
        for(int i=0;i<m;i++){
            b[i] = rand(-1.0,0);

        }
        return b;
    }
    private void init_array(double num,double[] array){
        for(int i=0;i<array.length;i++){
            array[i] =num;
        }
    }
    private void init_array(int num,int[] array){
        for(int i=0;i<array.length;i++){
            array[i] =num;
        }
    }
    private double[] get_shift(double[] score){
        int index = get_max_index(score);
        double max = score[index];
        double[] return_score = new double[score.length];
        for(int i=0;i<score.length;i++){
            return_score[i] = score[i] - max;
        }
        return return_score;
    }
    private int get_max_index(double[] array){
        int index = 0;
        for(int i=0;i<array.length;i++){
            if(array[i]>array[index]){
                index = i;
            }
        }
        return index;
    }



    private double[] softmax(double[] x){
        double[] x_shift = get_shift(x);
        double[] result = new double[x.length];
        double sum = 0;
        for(int i=0;i<x.length;i++){
            sum+=Math.exp(x_shift[i]);
        }
        for(int i=0;i<x.length;i++){
            result[i] = Math.exp(x_shift[i])/sum;
        }

        return result;
    }

    private double tanh(double x){
        return Math.tanh(x);
    }

    private double tanh_deriv(double x) {
        double result = 1-tanh(x)*tanh(x);
        return result;

    }

    public BPNetwork(int input_n, int output_n, int[] hidden_set){
        this.input_n = input_n +1;
        this.output_n = output_n;
        this.hidden_ns = new int[hidden_set.length];
        for(int i=0;i<hidden_set.length;i++) {
            this.hidden_ns[i] = hidden_set[i] + 1;
        }

        this.input_cells = new int[this.input_n];
        init_array(1,this.input_cells);

        this.output_cells = new double[this.output_n];
        init_array(1,this.output_cells);
        //初始化weights和bias
        this.input_w = generate_w(this.input_n, this.hidden_ns[0]);
        int hidden_length = this.hidden_ns.length;
        this.hidden_ws = new double[hidden_length-1][][];
        for(int i=0;i<hidden_length-1;i++){
            this.hidden_ws[i] = generate_w(this.hidden_ns[i],this.hidden_ns[i+1]);
        }

        this.output_w = generate_w(this.hidden_ns[hidden_length-1], this.output_n);
        this.output_b = generate_b(this.output_n);
        this.hidden_bs =new double[hidden_length][];
        for(int i=0;i<hidden_length;i++){
            this.hidden_bs[i] = generate_b(this.hidden_ns[i]);
        }

        this.hidden_results = new double[hidden_length][];
        this.predict = new double[this.output_n];
    }
    private double[] forward_proagate(int[] input){
        for(int i=0;i<input.length;i++){
            this.input_cells[i] = input[i];
        }
        //输入层
        this.hidden_results[0] = new double[this.hidden_ns[0]];
        for(int h=0;h<this.hidden_ns[0];h++){
            double total = 0;
            for(int i=0;i<this.input_n;i++){
                total+= this.input_w[i][h]*this.input_cells[i];
            }
            this.hidden_results[0][h] = tanh(total+this.hidden_bs[0][h]);
        }
        //隐藏层
        int last_hidden_index = this.hidden_ns.length-1;
        for(int k=0;k<last_hidden_index;k++) {
            this.hidden_results[k + 1] = new double[this.hidden_ns[k+1]];
            for(int h=0;h<this.hidden_ns[k+1];h++){
                double total = 0.0;
                for(int i=0;i<this.hidden_ns[k];i++){
                    total += this.hidden_ws[k][i][h] * this.hidden_results[k][i];
                }
                this.hidden_results[k+1][h] = tanh(total+ this.hidden_bs[k+1][h]);
            }
        }
        //输出层
        for(int h=0;h<this.output_n;h++){
            double total = 0.0;
            for(int i=0;i<this.hidden_ns[last_hidden_index];i++){
                total+=this.output_w[i][h] * this.hidden_results[last_hidden_index][i];
            }
            this.output_cells[h] = total;
        }
        this.predict = softmax(this.output_cells);
        return predict;
    }
    private void get_deltas(int label){
        //输出层deltas
        this.output_deltas = new double[this.output_n];
        init_array(0.0,this.output_deltas);
        for(int j=0;j<this.output_n;j++){
            if(label==j){
                this.output_deltas[j]+= 1-this.predict[j];
            }
            else {
                this.output_deltas[j] += 0-this.predict[j];
            }
        }
        //隐藏层deltas
        double[] tmp_deltas = this.output_deltas;
        double[][] tmp_weight = this.output_w;
        this.hidden_deltases = new double[this.hidden_ns.length][];
        for(int k = this.hidden_ns.length-1; k>=0; k=k-1){
            this.hidden_deltases[k] = new double[this.hidden_ns[k]];
            init_array(0,this.hidden_deltases[k]);
            for(int o=0;o<this.hidden_ns[k];o++){
                double error = 0;
                for(int i=0;i<tmp_deltas.length;i++){
                    error+=tmp_deltas[i] * tmp_weight[o][i];

                }
                this.hidden_deltases[k][o] = tanh_deriv(this.hidden_results[k][o]) * error;
            }
            if(k>0){
                tmp_weight = this.hidden_ws[k-1];
                tmp_deltas = this.hidden_deltases[k];
            }
            else
                break;
        }
    }

    private void renew_w(double learn){
        //更新隐藏层到输出层的权重
        double change;
        int k = this.hidden_ns.length-1;
        for(int i=0;i<this.hidden_ns[k];i++){
            for(int o=0;o<this.output_n;o++){
                change = this.output_deltas[o] * this.hidden_results[k][i];
                this.output_w[i][o] += change * learn;
            }
        }
        //更新隐藏层的权重
        while (k>0){
            for(int i=0;i<this.hidden_ns[k-1];i++){
                for(int o=0;o<this.hidden_ns[k];o++){
                    change = this.hidden_deltases[k][o] * this.hidden_results[k-1][i];
                    this.hidden_ws[k-1][i][o] += change *learn;
                }
            }
            k--;
        }
        //更新输入层到隐藏层的权重
        for(int i=0;i<this.input_n;i++){
            for(int o=0;o<this.hidden_ns[0];o++){
                change = this.hidden_deltases[0][o] * this.input_cells[i];
                this.input_w[i][o] += change * learn;
            }
        }
    }
    private void renew_b(double learn){
        for(int k=this.hidden_bs.length-1;k>=0;k--){
            for(int i=0;i<this.hidden_ns[k];i++){
                this.hidden_bs[k][i] = this.hidden_bs[k][i] + learn * this.hidden_deltases[k][i];
            }
        }
    }

    private double back_propagate(int[] input, int label, double learn){
        this.forward_proagate(input);
        this.get_deltas(label);
        this.renew_w(learn);
        this.renew_b(learn);
        return get_loss(label);
    }
    private double get_loss(int label){
        double loss = 0;
        double[] score_shift = get_shift(this.output_cells);
        double sum = 0;
        for(int i=0;i<score_shift.length;i++){
            sum+=Math.exp(score_shift[i]);
        }
        loss = Math.log (sum)- score_shift[label];
        return loss;
    }
    public double get_average_test_loss(){
        double loss = 0;
        for(int i=0;i<test_label.length;i++){
            forward_proagate(test_datas[i]);
            loss+= get_loss(test_label[i]);
        }
        loss = loss/test_label.length;
        return loss;
    }

    public void train(double learn,int limit){
        double error;
        for(int j=0;j<limit;j++){
            error = 0;
            for(int i=0;i<train_datas.length;i++){
                int[] input = train_datas[i];
                int label = train_label[i];
                error += this.back_propagate(input,label,learn);
            }
          //  System.out.println(error);
            if(j % 4==0){
                test_result();
                train_test();
            }
           upset_train_data();
        }


    }

    public double train_test() {
        double count = 0;
        for(int i=0;i<train_datas.length;i++){
            double[] test_result = forward_proagate(train_datas[i]);
            if(get_max_index(test_result)==train_label[i])
                count= count+1;
        }
        double correction = count/train_label.length;
        return correction;
        //
    }

    public double test_result() {
       // System.out.println("-----test_data------");
        double count = 0;

        for(int i=0;i<test_datas.length;i++){
            double[] test_result = forward_proagate(test_datas[i]);
            if(get_max_index(test_result)==test_label[i])
                count= count+1;
        }
        double correction = count/test_label.length;
        return correction;
        //
    }
    public void upset_train_data(){
        int train_len = train_datas.length;
        int index = (int)rand(0,train_len);
        int index_mid = index/2;
        for(int i=0;i<index_mid;i++){
            int[] tmp_data = train_datas[index_mid+i];
            int tmp_label = train_label[index_mid+i];
            train_datas[index_mid+i] = train_datas[i];
            train_label[index_mid+i] = train_label[i];
            train_label[i] = tmp_label;
            train_datas[i] = tmp_data;
        }
        int remain_mid = (train_len-index)/2;
        for(int i=0;i<remain_mid;i++){
            int[] tmp_data = train_datas[index + i];
            train_datas[index + i] = train_datas[index + remain_mid + i];
            train_datas[index + remain_mid + i] = tmp_data;
            int tmp_label = train_label[index + i];
            train_label[index + i] = train_label[index + remain_mid + i];
            train_label[index + remain_mid + i] = tmp_label;
        }
    }

    public void test() {
        System.out.println("-----train-data-----");
        this.train(0.06,150);
        System.out.println("训练集正确率为：" + train_test());
        System.out.println("测试集正确率为：" + test_result());
    }


}
