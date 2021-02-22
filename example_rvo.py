import numpy as np

##from policies.SLSTMPolicy import SLSTMPolicy
##policy = SLSTMPolicy()
##policy.init()


from policies.RVOPolicy import RVOPolicy
policy = RVOPolicy()
policy.init()

data = np.array(
        [[[12.91624378, 10.83968944],
        [12.83100541, 10.51081624],
        [12.74787169, 10.17716984],
        [12.68473216,  9.80342859],
        [12.62159262,  9.42944869],
        [12.55466472,  9.05857136],
        [12.45364147,  8.71490125],
        [12.35261821,  8.37123113]],

       [[13.55100655, 10.8215513 ],
        [13.46829376, 10.38671314],
        [13.38558097,  9.95211365],
        [13.30139493,  9.5623822 ],
        [13.21720888,  9.17241208],
        [13.13302284,  8.78268063],
        [12.95875773,  8.42063371],
        [12.78470308,  8.0585868 ]],

       [[ 5.06694751,  4.39921609],
        [ 5.75095912,  4.55840217],
        [ 6.43497073,  4.71758826],
        [ 7.09267419,  4.87080785],
        [ 7.72406952,  5.01806095],
        [ 8.35546485,  5.1655527 ],
        [ 8.99106948,  5.29228106],
        [ 9.66455783,  5.23261611]],

       [[ 2.3875162 ,  5.67222746],
        [ 2.87095456,  5.77819241],
        [ 3.31356269,  5.86840582],
        [ 3.74627895,  5.954562  ],
        [ 4.17899522,  6.04071819],
        [ 4.60792311,  6.10825891],
        [ 5.02211845,  6.1006218 ],
        [ 5.43652425,  6.09322335]],

       [[ 1.09631276,  8.17027958],
        [ 1.39243717,  7.96908937],
        [ 1.69213948,  7.81133524],
        [ 2.0008918 ,  7.75572751],
        [ 2.30943365,  7.69988112],
        [ 2.64428364,  7.65023988],
        [ 3.03953711,  7.61396359],
        [ 3.43500105,  7.57792596]],

       [[ 0.68211742,  7.87410277],
        [ 0.98371392,  7.71945122],
        [ 1.31645926,  7.59630277],
        [ 1.64899413,  7.47315431],
        [ 1.99415691,  7.36623472],
        [ 2.38877899,  7.32446925],
        [ 2.78340107,  7.28270379],
        [ 3.17802316,  7.24093832]],

       [[14.04244258,  3.7906336 ],
        [13.49081353,  3.67536092],
        [12.82427052,  3.6992269 ],
        [12.15793799,  3.72309288],
        [11.49139499,  3.74695886],
        [10.89009617,  3.70614803],
        [10.28879735,  3.66509854],
        [ 9.68749853,  3.62428772]],

       [[ 1.17734182,  4.12881454],
        [ 1.71949995,  4.12881454],
        [ 2.27912667,  4.14528206],
        [ 2.85032898,  4.17248928],
        [ 3.42153129,  4.1996965 ],
        [ 3.99946848,  4.23263155],
        [ 4.63654637,  4.31640114],
        [ 5.27362425,  4.40040939]],

       [[ 0.93951625,  4.88918466],
        [ 1.49577553,  4.85505631],
        [ 2.03582901,  4.87438775],
        [ 2.56914759,  4.91686919],
        [ 3.10225572,  4.95935064],
        [ 3.63536384,  5.00183208],
        [ 4.19267545,  5.04670012],
        [ 4.84553822,  5.10183054]],

       [[14.51514722,  4.37535011],
        [13.87659607,  4.40733052],
        [13.2382554 ,  4.43907227],
        [12.62853797,  4.4555398 ],
        [12.01903102,  4.47200733],
        [11.41099731,  4.4846563 ],
        [10.81790663,  4.46293825],
        [10.22481596,  4.44122021]]])

agent_index = 1
#result = policy.predict(data, agent_index)
goal= np.array([11,6])
result = policy.predict(data, agent_index, goal, pref_speed= 1.0)  


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



name = "visualization"
metadata = dict(title=name, artist='Sam Shum',
                comment='visualization')
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

                patches.append(ax.add_patch( plt.Circle((records[i][0]+0.2,records[i][1]+0.2),0.2,color=[0,0,0,((i/total_history_len)+0.05 ) ] ,linewidth=0.00001      )))
                    
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


