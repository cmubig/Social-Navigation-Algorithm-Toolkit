import numpy as np

from policies.SLSTMPolicy import SLSTMPolicy
policy = SLSTMPolicy()
policy.init()


##from policies.SOCIALFORCEPolicy import SOCIALFORCEPolicy
##policy = SOCIALFORCEPolicy()
##policy.init()

data = np.array([[[ 8.44,  7.05],
        [ 8.44,  7.05],
        [ 8.44,  7.05],
        [ 8.44,  7.05],
        [ 8.44,  7.05],
        [ 8.44,  7.05],
        [ 8.44,  7.05],
        [ 8.44,  7.05]],

       [[ 8.84,  8.09],
        [ 8.84,  8.09],
        [ 8.84,  8.09],
        [ 8.84,  8.09],
        [ 8.84,  8.09],
        [ 8.84,  8.09],
        [ 8.84,  8.09],
        [ 8.84,  8.09]],

       [[ 3.35, 12.32],
        [ 3.22, 12.42],
        [ 3.21, 12.49],
        [ 3.36, 12.65],
        [ 3.45, 12.79],
        [ 3.47, 12.79],
        [ 3.38, 12.79],
        [ 3.16, 12.78]]])

agent_index = 0
result = policy.predict(data, agent_index)

goal= np.array([11,6])
#result = policy.predict(data, agent_index, goal, pref_speed= 1.0)  


#plot
from matplotlib import pyplot as plt
from matplotlib import animation

import matplotlib
matplotlib.rcParams.update({'font.size': 13})

plt_colors = []
plt_colors.append([0.8500, 0.3250, 0.0980])  # orange
plt_colors.append([0.0, 0.4470, 0.7410])  # blue
plt_colors.append([0.4660, 0.6740, 0.1880])  # green
plt_colors.append([0.4940, 0.1840, 0.5560])  # purple
plt_colors.append([0.9290, 0.6940, 0.1250])  # yellow
plt_colors.append([0.3010, 0.7450, 0.9330])  # cyan
plt_colors.append([0.6350, 0.0780, 0.1840])  # chocolate

def color(agent_id, decay=None):
    if decay is None:
        return plt_colors[agent_id%len(plt_colors)]
    else:
        print(plt_colors[agent_id%len(plt_colors)])
        print(decay)
        color_with_alpha = plt_colors[agent_id%len(plt_colors)].copy()
        color_with_alpha.append(decay)
        return color_with_alpha

fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(8, 8)

x_min = -3  #-5   #-1 17 for univ   #-5 15 for biwi eth  #-3 17 for zara1  #biwi_hotel -12 8
x_max = 17   #15
y_min = -3  #-5
y_max = 17   #15

plt.axis([x_min,x_max,y_min,y_max]) #for simulator
#plt.axis([0,20,0,20]) #for biwi_eth

ax = plt.gca()
# Major ticks every 20, minor ticks every 5
major_ticks = np.arange(x_min, x_max+1, 1)
minor_ticks = np.arange(x_min, x_max+1, 0.2)

ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.minorticks_on()

# Customize the major grid
ax.grid(which='major', linestyle='-', linewidth='0.5', color='red'   , alpha=0.2)
# Customize the minor grid
ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black' , alpha=0.2)



name = "visualization_slstm"
metadata = dict(title=name, artist='',
                comment=name)
writer_mp4 = animation.FFMpegFileWriter(fps=(3), metadata=metadata)
writer_gif = animation.ImageMagickFileWriter(fps=(3), metadata=metadata)

prediction_step = 8 # 0-7 data    8 =prediction use

def init():
    return []




def animate(i):
    if i!=prediction_step:
        patches = []
        ax.patches = []
        ax.annotations = []
        records = data[:,:(i+1)]
        #print(records)
        for agent in range(len(records)):

            #since we want to show current as well as past points
            total_history_len = len(records[agent][:,0])

            print("Agent "+str(agent))
            for i in range(total_history_len):

                patches.append(ax.add_patch( plt.Circle((records[agent][:,0][i]+0.2,records[agent][:,1][i]+0.2),0.2,color=color(agent,  ((i/total_history_len)+0.1 )) ,linewidth=0.00001      )))
                
        patches.append(ax.legend(["Timestep "+str(i*10).rjust(6)],loc="upper right", prop={'size': 14},handlelength=0.00001, handletextpad=0.00001, markerfirst=False))

            
        print(i)

        return patches


    else:
        patches = []
        records = result
        #print(records)
        #since we want to show current as well as past points
        total_history_len = len(records)

        print("Agent "+str(agent_index))

        if len(result.shape)==1:
            patches.append(ax.add_patch( plt.Circle((records[0]+0.2,records[1]+0.2),0.2,color=[0,0,0,1 ] ,linewidth=0.00001      )))
                
            patches.append(ax.legend(["Prediction".rjust(6)],loc="upper right", prop={'size': 14},handlelength=0.00001, handletextpad=0.00001, markerfirst=False))
            
        else:
            for i in range(total_history_len):

                patches.append(ax.add_patch( plt.Circle((records[i][0]+0.2,records[i][1]+0.2),0.2,color=[0,0,0,(1-(i/total_history_len)+0.05 ) ] ,linewidth=0.00001      )))
                    
            patches.append(ax.legend(["Prediction".rjust(6)],loc="upper right", prop={'size': 14},handlelength=0.00001, handletextpad=0.00001, markerfirst=False))

            
            
        print(i)

        return patches

        


anim = animation.FuncAnimation(fig, animate,
                               init_func=init,
                               frames=9,  #total 9 frames => 8 timestep + 1 timestep for showing prediction
                               interval=333, #ms for each frame
                               blit=True,
                               repeat=False)

anim.save(str(name)+'.mp4', writer=writer_mp4)


