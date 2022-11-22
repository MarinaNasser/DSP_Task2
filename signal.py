def signal_modification(points_per_freq,max_freq,sliders_num,amplitude,sliders_data,mode,ranges):
    empty = st.empty()
    empty.empty()
    
    if mode==2 or mode==1:
        target_freq=music_fn(ranges)
    #     for instrumentIndex in range(len(ranges)):
    #         for index in range(0,len(ranges[instrumentIndex]),2):
                # amplitude[int(ranges[instrumentIndex][index]*points_per_freq):int(ranges[instrumentIndex][index+1]*points_per_freq)]*=sliders_data[instrumentIndex] 
    else:
        target_freq=uniform_fn(max_freq,sliders_num)
    for i in range(0,sliders_num,2):  
        amplitude[int(target_freq[i]*points_per_freq) :int(target_freq[i+1]*points_per_freq)]*=sliders_data[i]
    
   
    return amplitude,empty  

def uniform_fn(max_freq,sliders_num):
    list=[]
    target=int(max_freq/sliders_num)
    for i in range(0,sliders_num): 
         list.append(int(target*i))
         list.append(int(target*(i+1)))
    return list

def music_fn(ranges):
    list=[]
    for instrumentIndex in range(len(ranges)):
        for index in range(0,len(ranges[instrumentIndex]),2):
            list.append(int(ranges[instrumentIndex][index]))
            list.append(int(ranges[instrumentIndex][index+1]))
    return list