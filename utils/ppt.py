'''
gesture 감지 -> 로직 -> 조작
'''

def output_to_detect(s):
    '''
    감지된 gesture를 리스트로 반환한다.
    감지되지 않으면 빈 리스트를 반환한다.
    '''
    temp = s.split()
    label_dict = ['K', 'L', 'paper', 'rock', 'scissor', 'W']
    send_result = []
    
    if len(temp) > 2:
        label_list = temp[3::2]
        label_num = temp[2::2]
        # print(label_list, label_num)
        for n, label in zip(label_num, label_list):
            send_result += [label_dict[int(label[0])]] * int(float(n))
    # print(send_result)
    return send_result