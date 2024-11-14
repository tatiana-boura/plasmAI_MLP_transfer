%Data splitting code

dataTable = readmatrix('merged_data_1-8719.csv'); 

%test and train sizes 
test_size = 0.2; 
train_size = 1-test_size; 

Area_of_data = (max(dataTable(:,2))-min(dataTable(:,2)))*(max(dataTable(:,1))-min(dataTable(:,1))); 

% Calculate the side length of the square based on test area
test_area = test_size*Area_of_data;
side_length_power = 700;
side_length_pressure = test_area/side_length_power;

%Choose the center of the square 
center_Pressure = max(dataTable(:,2));
center_Power = max(dataTable(:,1));

%Corners 
x_min = center_Power - side_length_power;
x_max = center_Power;

y_min = center_Pressure - side_length_pressure;
y_max = center_Pressure;

%find the data points within the square 
test_data=[];
train_data=[];; 
k=0;
h=0;
for i=1:length(dataTable(:,1)) %power
        if (dataTable(i,1) >= x_min) && (dataTable(i,1) <= x_max) && (dataTable(i,2) >= y_min) && (dataTable(i,2) <= y_max)
            k=k+1;
            test_data(k,:)=dataTable(i,:);
        else 
            h=h+1;
            train_data(h,:)=dataTable(i,:);
        end    
end



hold on 
plot(train_data(:,1),train_data(:,2), 'bo');
plot(test_data(:,1),test_data(:,2), 'ro');
legend('Train Set', 'Test Set');
xlabel('Power'); ylabel('Pressure'); title('data-grid');
hold off;

% Save test data to a semicolon-delimited CSV file
writematrix(test_data, 'test_data_no_head_outer_corner.csv', 'Delimiter', ';');
writematrix(train_data, 'train_data_no_head_outer_corner.csv', 'Delimiter', ';');

