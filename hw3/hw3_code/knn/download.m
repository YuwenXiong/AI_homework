urlwrite('http://jwbinfosys.zju.edu.cn/CheckCode.aspx', 'code.png');
tX = extract_image('code.png');
show_image(tX);
