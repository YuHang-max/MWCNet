from typing import Union
import time


def num_2_str(num: Union[float, int]):
    if num / 1e5 >= 1:
        if isinstance(num, int):
            return f'  {num} '
        return f'{num:.4f}'
    elif num / 1e4 >= 1:
        if isinstance(num, int):
            return f'   {num} '
        return f' {num:.4f}'
    elif num / 1e3 >= 1:
        if isinstance(num, int):
            return f'    {num} '
        return f'  {num:.4f}'
    elif num / 1e2 >= 1:
        if isinstance(num, int):
            return f'     {num} '
        return f'   {num:.4f}'
    elif num / 1e1 >= 1:
        if isinstance(num, int):
            return f'      {num} '
        return f'    {num:.4f}'
    else:
        if isinstance(num, int):
            return f'       {num} '
        return f'     {num:.4f}'


def show_title(*args, **kwargs) -> None:
    print(
        '======================================================================================================='
        '============================================')
    print(
        '|    Epoch   |     Loss     || Test Epoch |     Loss     |      SAM     |     RMSE     |     PSNR     ||'
        '     Loss     |  Max Epoch |   Max PSNR   |')


def showTrainInfo(
        epoch: int = 0, loss: float = 0.0, test_epoch: int = 0, test_loss: float = 0.0, sam: float = 0.0,
        rmse: float = 0.0, psnr: float = 0.0, max_epoch: int = 0, max_psnr: float = 0.0, max_loss: float = 0.0
) -> bool:
    print('\r|', end='')
    print(num_2_str(epoch), end='   |')
    print(num_2_str(loss), end='   ||')
    print(num_2_str(test_epoch), end='   |')
    print(num_2_str(test_loss), end='   |')
    print(num_2_str(sam), end='   |')
    print(num_2_str(rmse), end='   |')
    print(num_2_str(psnr), end='   ||')
    print(num_2_str(max_loss), end='   |')
    print(num_2_str(max_epoch), end='   |')
    print(num_2_str(max_psnr), end='   |')
    return True


if __name__ == '__main__':
    show_title()
    for i in range(14):
        showTrainInfo(epoch=i, loss=float(i ** 2))
        time.sleep(1)
