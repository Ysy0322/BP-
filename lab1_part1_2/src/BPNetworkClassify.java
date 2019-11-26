public class BPNetworkClassify {

    public static void set_data_label(BPNetwork bpNetwork,double rate){
        ReadImages readImages = new ReadImages();
        readImages.set_data_label(rate);
        bpNetwork.train_datas = readImages.getTrain_data();
        bpNetwork.train_label = readImages.getTrain_label();
        bpNetwork.test_datas = readImages.getTest_data();
        bpNetwork.test_label = readImages.getTest_label();
    }


    public static void main(String[] are){
        int[] hidden_set = {150};
        BPNetwork bpNetwork = new BPNetwork(784,14,hidden_set);
        set_data_label(bpNetwork,0.8);
        bpNetwork.train(0.06,150);
        System.out.println(bpNetwork.get_average_test_loss());
        System.out.println(bpNetwork.test_result());
    }
}
