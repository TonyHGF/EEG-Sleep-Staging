import pyedflib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import matplotlib.patches as mpatches

def read_result_file(file_path):
    with pyedflib.EdfReader(file_path) as reader:
        annotations = reader.readAnnotations()
        return annotations[0], annotations[1], annotations[2]
    
def plot_sleep_stages(start_times, durations, labels):
    # 为每个睡眠阶段分配颜色
    color_dict = {
        'Sleep stage 1': 'blue',
        'Sleep stage 2': 'green',
        'Sleep stage 3': 'yellow',
        'Sleep stage 4': 'orange',
        'Sleep stage R': 'purple',
        'Sleep stage W': 'red',
        'Sleep stage ?': 'grey'
    }

    # 准备绘图
    plt.figure(figsize=(10, 2))
    base_time = datetime.datetime.now()

    # 绘制每个阶段
    for start, duration, label in zip(start_times, durations, labels):
        color = color_dict.get(label, 'grey')
        start_datetime = base_time + datetime.timedelta(seconds=start)
        end_datetime = start_datetime + datetime.timedelta(seconds=duration)
        plt.hlines(1, start_datetime, end_datetime, colors=color, lw=10)

    handles = [mpatches.Patch(color=color, label=label) for label, color in color_dict.items()]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')


    # 设置图表格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gca().yaxis.set_visible(False)  # 隐藏y轴
    plt.gcf().autofmt_xdate()
    plt.xlabel('time')
    plt.title('Sleep Stages')
    plt.tight_layout()
    plt.show()