import serial
from serial import Serial
import time
import cv2 as cv



def raplar(img, ChuanKou):

    ser = serial.Serial(ChuanKou, 256000, timeout=5)
    ser.flushInput()

    size = img.shape[1]
    data = 'A5 82 05 00 00 00 00 00 22'
    data = bytes.fromhex(data)

    ser.write(data)
    time.sleep(1)
    count = ser.inWaiting()
    # print(count)
    ser.flushInput()
    while True:

        count = ser.inWaiting()  # 获取串口缓冲区数据
        # print(count)
        time.sleep(1)
        if count > 0:
            recv = ser.read(count)  # # 读出串口数据，数据采用gbk编码

            print(recv)  # 打印一下子

            time.sleep(1)
            # 延时0.1秒，免得CPU出问题
            ser.flushInput()
            R = recv.split()
            print(R)
            a = float(R[0].decode(encoding='utf-8')) * 0.001
            # a = "% m" %a
            b = float(R[1].decode(encoding='utf-8')) * 0.001
            # a = "% m" %a
            # a = "% m" %b
            c = float(R[2].decode(encoding='utf-8')) * 0.001
            d = int(200 - (10 * (a - 1)))
            e = int(200 - (10 * (b - 1)))
            f = int(200 - (10 * (c - 1)))
            # a = "% m" %a
            # a = "% m" %c
            # print(a, '+', b, '+', c)
            cv.line(img, (0, e), (int(size/3), e), (0, 255, 255), thickness=1, )
            cv.line(img, (int(size/3), d), (int(2*size/3), d), (0, 255, 0), thickness=1, )
            cv.line(img, (int(2*size/3), f), (size, f), (0, 0, 255), thickness=1, )
            cv.putText(img, '%.2f m' % b, (20, e - 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv.putText(img, '%.2f m' % a, (360, d - 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv.putText(img, '%.2f m' % c, (710, f - 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            # cv.imshow("Canvas", img)

            # cv.waitKey(0)
            #cv.destroyAllWindows()
            ser.flushInput()

if __name__ == '__main__':

    raplar('v_47.png', 'COM4')
