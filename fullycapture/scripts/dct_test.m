joint_mat = load('./pose6_171204_raw.txt');
joint_median = load('./pose6_171204_median.txt');



%joint_25
%vis sth.
joint_list = [25];
N_joint = length(joint_list);

x0 = [];
x1 = [];

for i=1:N_joint
    jid = joint_list(i);
    xi = joint_mat(:,[jid*4-3,jid*4-2,jid*4-1]);
    x0 = cat(2,x0,xi);
    xi = joint_median(:,[jid*4-3,jid*4-2,jid*4-1]);
    x1 = cat(2,x1,xi);
end

x11 = medfilt1(x0);

%try with several naive, simple filtering strategy.
figure(1);
plot(x0);
hold on;
plot(x1);


%prior-enforced and re-weighted filtering.
