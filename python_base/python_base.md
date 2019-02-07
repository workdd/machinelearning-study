
# 사칙 연산


```python
1+2
```




    3




```python
(1+2)*(4/2)
```




    6.0



# 제곱


```python
2**8
```




    256



# 변수


```python
x =1
y =1/3
x+y
```




    1.3333333333333333



# 자료형

int float str bool list tuple ndarray


```python
type(100)
```




    int




```python
type(100.0)
```




    float




```python
x = '123'
type(x)
```




    str



# print 문


```python
x =1/3
x
y = 2/3
y
```




    0.6666666666666666




```python
x =1/3
print(x)
y = 2/3
print(y)
```

    0.3333333333333333
    0.6666666666666666
    


```python
print('x는?'+str(x)+'입니다.')
```

    x는?0.3333333333333333입니다.
    


```python
print('x는 {0}'.format(x)+'입니다.')
```

    x는 0.3333333333333333입니다.
    


```python
x =1
y =2
z =3
print('x는 {0} y는 {1} z는 {2}'.format(x,y,z))
```

    x는 1 y는 2 z는 3
    

# list


```python
x = [1,2,3]
print(x)
print(type(x))
```

    [1, 2, 3]
    <class 'list'>
    


```python
print(x[0],x[1],x[2])
print(type(x[0]))
```

    1 2 3
    <class 'int'>
    


```python
s =['a',1,'b',2]
print(s[0],type(s[0]))
print(s[1],type(s[1]))
```

    a <class 'str'>
    1 <class 'int'>
    

2차원 배열


```python
a = [[1,2,3],[4,5,6]]
print(a)
print(a[0])
print(a[1][1])
```

    [[1, 2, 3], [4, 5, 6]]
    [1, 2, 3]
    5
    

리스트 수정


```python
a = [[1,2,3],[4,5,6]]
print(a)
a[0][1] = 100
a[1] = [1000.1001,1002]
print(a)
```

    [[1, 2, 3], [4, 5, 6]]
    [[1, 100, 3], [1000.1001, 1002]]
    

list 길이


```python
print(len(a))
print(len(a[0]))
```

    2
    3
    

연속된 정수 데이터 작성


```python
x = range(5,10)
print(x[0],x[1],x[2],x[3],x[4])
```

    5 6 7 8 9
    


```python
print(x[5])
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-29-394df876e9e9> in <module>
    ----> 1 print(x[5])
    

    IndexError: range object index out of range



```python
print(x)
print(type(x))
```

    range(5, 10)
    <class 'range'>
    


```python
z = list(range(5,10))
print(z)
print(type(z))
```

    [5, 6, 7, 8, 9]
    <class 'list'>
    


```python
list(range(10))
```




    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]



# tuple


```python
a = (1,2,3)
print(a)
print(type(a))
```

    (1, 2, 3)
    <class 'tuple'>
    


```python
print(a[1])
```

    2
    


```python
a[1] =100
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-39-72a917d5d7d0> in <module>
    ----> 1 a[1] =100
    

    TypeError: 'int' object does not support item assignment


길이가 1인 tuple 선언


```python
a = (1)
print(type(a))
b = (1,)
print(type(b))
```

    <class 'int'>
    <class 'tuple'>
    

# if


```python
x = 10
if x>5:
    print('x is larger than 5')
elif x<3:
    print('x is smaller than 3')
else:
    print('x는 3과 5사이 어딘가')

```

    x is larger than 5
    

비교 연산자


```python
x =10
print(x>5)
print(x<3)
print(type(x>5))
```

    True
    False
    <class 'bool'>
    


```python
a =10
b = 9
print(type(a==b))
print(type(a!=b))
print(type(a>=b))
print(type(a>b))
print((a==b))
print((a!=b))
print((a>=b))
print((a>b))
```

    <class 'bool'>
    <class 'bool'>
    <class 'bool'>
    <class 'bool'>
    False
    True
    True
    True
    


```python
x =15
if 10<=x and x>20:
    print('x is between 10 and 20')
else:
    print('x is not between 10 and 20')
y = 10
if 5<=y or y >=10:
    print('y is 5<= or y>=10')
```

    x is not between 10 and 20
    y is 5<= or y>=10
    

# for문


```python
for i in [1,2,3]:
    print(i)
```

    1
    2
    3
    


```python
num = [1,2,3,4,5]
for i in range(len(num)):
    num[i] = num[i]+1
print(num)
```

    [2, 3, 4, 5, 6]
    

enumerate


```python
num =[1,2,3,4,5]
for i,n in enumerate(num):
    num[i] = n+2
print(num)
```

    [3, 4, 5, 6, 7]
    

# 벡터

일반적인 list


```python
[1,2]+[3,4]
```




    [1, 2, 3, 4]



numpy 이용


```python
import numpy as np
x = np.array([1,2,3])
print(x)
print(type(x))
```

    [1 2 3]
    <class 'numpy.ndarray'>
    


```python
print(x[0])
x[0] =100
print(x)
```

    100
    [100   2   3]
    


```python
print(np.arange(10))
print(np.arange(5,10))
```

    [0 1 2 3 4 5 6 7 8 9]
    [5 6 7 8 9]
    

ndarray 형의 주의점


```python
a = np.array([1,1])
b =a
print(a)
print(b)
b[0]= 100
print(a)
print(b)
```

    [1 1]
    [1 1]
    [100   1]
    [100   1]
    


```python
a = np.array([1,1])
b = a.copy()
print(a)
print(b)
b[0]=100
print(a)
print(b)
```

    [1 1]
    [1 1]
    [1 1]
    [100   1]
    

# 행렬

ndarray의 2차원 배열로 행렬을 정의


```python
x = np.array([[1,2,3],[4,5,6]])
print(x)
```

    [[1 2 3]
     [4 5 6]]
    


```python
print(x.shape)
w,h = x.shape
print(w,h)
```

    (2, 3)
    2 3
    


```python
print(x[1,2])
```

    6
    


```python
print(x[1][2])
```

    6
    


```python
x[1,2] =100
print(x)
```

    [[  1   2   3]
     [  4   5 100]]
    


```python
print(np.zeros(10))
print(np.zeros((2,10)))
print(np.ones((2,10)))
```

    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
    [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]
    


```python
np.random.rand(2,3)
```




    array([[0.494728  , 0.21410674, 0.30827843],
           [0.7150299 , 0.3766618 , 0.37584576]])




```python
np.random.randn(2,3)
```




    array([[-0.25814895, -1.30873281,  1.78920872],
           [-0.28845718, -0.24640404,  0.65188733]])




```python
np.random.randint(1,10,(2,3))
```




    array([[7, 3, 4],
           [2, 6, 7]])




```python
a = np.arange(10)
print(a)
```

    [0 1 2 3 4 5 6 7 8 9]
    


```python
a.reshape(2,5)
```




    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])




```python
a.reshape(2,4)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-75-5fcd0a049a15> in <module>
    ----> 1 a.reshape(2,4)
    

    ValueError: cannot reshape array of size 10 into shape (2,4)



```python
a.reshape(3,4)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-76-47db8ded5120> in <module>
    ----> 1 a.reshape(3,4)
    

    ValueError: cannot reshape array of size 10 into shape (3,4)


행렬의 사칙연산


```python
x = np.array([[1,1,1],[2,2,2]])
y= np.array([[2,2,2],[3,3,3]])
print(x+y)
```

    [[3 3 3]
     [5 5 5]]
    


```python
print(10*x)
```

    [[10 10 10]
     [20 20 20]]
    


```python
x = np.array([[1.4,1.4,1.4],[2,2,2]])
print('지수함수:',np.exp(x))
print('로그함수:',np.log(x))
print('평균:',np.mean(x))
print('표준편차:',np.std(x))
print('최댓값',np.max(x))
print('최솟값',np.min(x))
print('반올림:',np.round(x))
```

    지수함수: [[4.05519997 4.05519997 4.05519997]
     [7.3890561  7.3890561  7.3890561 ]]
    로그함수: [[0.33647224 0.33647224 0.33647224]
     [0.69314718 0.69314718 0.69314718]]
    평균: 1.7
    표준편차: 0.30000000000000004
    최댓값 2.0
    최솟값 1.4
    반올림: [[1. 1. 1.]
     [2. 2. 2.]]
    


```python
v = np.array([[1,1,1],[2,2,2]])
w = np.array([[1,1],[2,2],[3,3]])
print(v.dot(w))
```

    [[ 6  6]
     [12 12]]
    

# 슬라이싱


```python
x = np.arange(10)
print(x)
print(x[:5])
print(x[3:])
print(x[1:5])
```

    [0 1 2 3 4 5 6 7 8 9]
    [0 1 2 3 4]
    [3 4 5 6 7 8 9]
    [1 2 3 4]
    


```python
print(x[::-1])
```

    [9 8 7 6 5 4 3 2 1 0]
    


```python
y = np.array([[1,1,1],[2,2,2],[3,3,3]])
print(y)
print(y[:2,1:])
```

    [[1 1 1]
     [2 2 2]
     [3 3 3]]
    [[1 1]
     [2 2]]
    

조건을 만족하는 데이터 수정


```python
x = np.arange(10)
print(x>3)
```

    [False False False False  True  True  True  True  True  True]
    


```python
print(x[x>3])
```

    [4 5 6 7 8 9]
    


```python
x[x>3]= 1000
print(x)
```

    [   0    1    2    3 1000 1000 1000 1000 1000 1000]
    

help


```python
help(np.random.randint)
```

    Help on built-in function randint:
    
    randint(...) method of mtrand.RandomState instance
        randint(low, high=None, size=None, dtype='l')
        
        Return random integers from `low` (inclusive) to `high` (exclusive).
        
        Return random integers from the "discrete uniform" distribution of
        the specified dtype in the "half-open" interval [`low`, `high`). If
        `high` is None (the default), then results are from [0, `low`).
        
        Parameters
        ----------
        low : int
            Lowest (signed) integer to be drawn from the distribution (unless
            ``high=None``, in which case this parameter is one above the
            *highest* such integer).
        high : int, optional
            If provided, one above the largest (signed) integer to be drawn
            from the distribution (see above for behavior if ``high=None``).
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        dtype : dtype, optional
            Desired dtype of the result. All dtypes are determined by their
            name, i.e., 'int64', 'int', etc, so byteorder is not available
            and a specific precision may have different C types depending
            on the platform. The default value is 'np.int'.
        
            .. versionadded:: 1.11.0
        
        Returns
        -------
        out : int or ndarray of ints
            `size`-shaped array of random integers from the appropriate
            distribution, or a single such random int if `size` not provided.
        
        See Also
        --------
        random.random_integers : similar to `randint`, only for the closed
            interval [`low`, `high`], and 1 is the lowest value if `high` is
            omitted. In particular, this other one is the one to use to generate
            uniformly distributed discrete non-integers.
        
        Examples
        --------
        >>> np.random.randint(2, size=10)
        array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0])
        >>> np.random.randint(1, size=10)
        array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
        Generate a 2 x 4 array of ints between 0 and 4, inclusive:
        
        >>> np.random.randint(5, size=(2, 4))
        array([[4, 0, 2, 1],
               [3, 2, 2, 0]])
    
    

# 함수


```python
def test():
    print('success!')
test()
```

    success!
    


```python
def plus(a,b):
    print(a+b)
plus(1,2)
```

    3
    


```python
def re():
    return 'hi'
re()
```




    'hi'




```python
def my_func(D):
    m = np.mean(D)
    s = np.std(D)
    return m,s
data = np.random.rand(100)
data_mean,data_std = my_func(data)
print('{0:3.2f},{1:3.2f}'.format(data_mean,data_std))
```

    0.55,0.29
    

# 파일 저장


```python
data = np.random.rand(2,3)
print(data)
np.save('datafile.npy',data)
data2 =np.load('datafile.npy')
print(data2)
```

    [[0.39236801 0.86777704 0.83055947]
     [0.3459329  0.60383957 0.05597821]]
    [[0.39236801 0.86777704 0.83055947]
     [0.3459329  0.60383957 0.05597821]]
    

여러 ndarray 형을 저장


```python
data1 = np.array([1,2,3])
data2 = np.array([10,20,30])
np.savez('datafile2.npz',data1=data1,data2=data2)
data1=[]
data2=[]
outfile =np.load('datafile2.npz')
print(outfile.files)

data1 = outfile['data1']
data2 = outfile['data2']
print(data1)
print(data2)
```

    ['data1', 'data2']
    [1 2 3]
    [10 20 30]
    
