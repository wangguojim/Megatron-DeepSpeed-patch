import os
import json
def split_data(path):
    for file in os.listdir(path):
        
        if 'dataset_10b_12.txt' in file:
            file_path=os.path.join(path,file)
            file_path1=os.path.join(path,'dataset_10b_12_1.txt')
            file_path2 = os.path.join(path, 'dataset_10b_12_2.txt')
            file_path3 = os.path.join(path, 'dataset_10b_12_3.txt')
            json_line = open(file_path, 'r', encoding='utf-8')
            json_line1 = open(file_path1, 'w', encoding='utf-8')
            json_line2 = open(file_path2, 'w', encoding='utf-8')
            json_line3 = open(file_path3, 'w', encoding='utf-8')
            i=0
            for line in json_line:
                if i % 3 == 0:
                    json_line1.write(line)
                if i % 3 == 1:
                    json_line2.write(line)
                if i % 3 == 2:
                    json_line3.write(line)
                i+=1
                # print(json.loads(line)['text'])
            json_line1.close()
            json_line2.close()
            json_line3.close()


def sub_data(input_path,output_path,samples=10000):

        json_line = open(input_path, 'r', encoding='utf-8')
        json_line1 = open(output_path, 'w', encoding='utf-8')

        i = 0
        for line in json_line:
            if i<samples:
                print(i)
                json_line1.write(line)
            else:
                break

            i += 1
            # print(json.loads(line)['text'])
        json_line1.close()



if __name__=='__main__':
    input_path='/nfs/dgx05/raid/guoqiang/dataset_10b_0.txt'
    out_path = '/nfs/dgx05/raid/guoqiang/aggregated_meta_data_test/dataset_10b_0.txt'
    sub_data(input_path, out_path,samples=2000)




