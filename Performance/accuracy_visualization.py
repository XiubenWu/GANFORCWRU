import xlrd
from matplotlib import pyplot as plt
import numpy as np

book = xlrd.open_workbook('accuracy/accuracy_dcwcgan.xls')
sheet = book.sheet_by_index(0)
nrows = sheet.nrows
ncols = sheet.ncols

# columu labels, gen num
col_label = []
for i in range(1, ncols):
    col_label.append(sheet.cell_value(0, i))
col_label = np.array(col_label, dtype=int)

# row labels, real num
row_label = []
for i in range(1, nrows):
    row_label.append(sheet.cell_value(i, 0))
# row_label = np.array(row_label, dtype=int)

# accuracy
accuracy = np.empty([nrows - 1, ncols - 1], dtype=float)
for i in range(1, nrows):
    for j in range(1, ncols):
        accuracy[i - 1][j - 1] = sheet.cell_value(i, j)

# plot
plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['figure.dpi'] = 10
# plt.rcParams['figure.figsize'] = (300, 300)
plt.title('DCWCGAN')
"""
SELFNOISEGAN  DCWCGAN
LetNet5  MobileNet  VGG16
"""
plt.xlabel('生成数据数量')
plt.ylabel('模型准确率')
for i in range(accuracy.shape[0]):
    plt.plot(col_label, accuracy[i, :], label=str(int(row_label[i])) + '个')
plt.legend(loc=4)
# plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
plt.savefig('visual.jpg', bbox_inches='tight')
plt.show()

print(sheet)
