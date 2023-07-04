% Basic setup
fclose('all');
NumDatum = 3;

% Create text sources
fs_root = ".\AO\SRC\";
ds = ".\datasetB12T.csv";

f_names = dir(fullfile(fs_root, '*.txt'));
file_handles = [];

for fhi = 1:length(f_names)
    file_handles(fhi,1) = fopen(fs_root + f_names(fhi).name, 'r', 'n', "UTF-8");
end

file_contents = {};

for fhi = 1:length(file_handles)
    file_contents{fhi,1} = [fgets(file_handles(fhi), inf)];
end

file_maxstart = strlength(file_contents)-386;

fopen(ds,"w");
fclose('all');

% Coders
bch_coders = {
    comm.BCHEncoder(31,26,bchgenpoly(31,26),6)
    comm.BCHEncoder(15,11,bchgenpoly(15,11),6)
    comm.BCHEncoder(7,4,bchgenpoly(7,4),4)
};

rs_coders = {
    comm.RSEncoder(15,8,"BitInput",true)
    comm.RSEncoder(15,12,"BitInput",true)
    comm.RSEncoder(7,4,"BitInput",true)
};


numgen = 1;
rindices = randi([1,7],1,NumDatum+1);
rstarts = rand(1,NumDatum+1);
rcodervar_h = randi([1,3],1,NumDatum+1);
rcodervar_b = randi([1,3],1,NumDatum+1);
rcodervar_r = randi([1,3],1,NumDatum+1);

while numgen <= NumDatum
    index = rindices(numgen);
    start = 1+floor(file_maxstart(index) * rstarts(numgen));
    utf8_string = file_contents{index}(start:start+384);
    numgen = numgen + 1;

    % Decode UTF-8 string into binary representation
    binary_string = dec2bin(utf8_string);
    code_string = reshape(binary_string',[],1);

    % Clip due to unpredictable lengths
    if length(code_string) < 3072
        binary_string = padarray(bin2dec(code_string), 3072-length(code_string), 0, 'post');
    elseif length(code_string) > 3072
        binary_string = bin2dec(code_string(1:3072));
    end
    
    % Write it raw
    writelines(sprintf('%d',binary_string(1:3072))+";0", ds, WriteMode="append");

    % Encode with Hamming code
    switch rcodervar_h(numgen)
        case 1
            % hamming_encoded = encode(binary_string, 255, 247, 'hamming/binary')';
            hamming_encoded = encode(binary_string, 31, 26, 'hamming/binary')';
        case 2
            hamming_encoded = encode(binary_string, 15, 11, 'hamming/binary')';
        case 3
            hamming_encoded = encode(binary_string, 7, 4, 'hamming/binary')';
        otherwise
            throw(MException('Illegal variant index %d', codevar))
    end

    hamming_str = sprintf('%d',hamming_encoded(1:3072))+";1";
    writelines(hamming_str, ds, WriteMode="append");

    % Encode with BCH code
    bch_code = bch_coders{rcodervar_b(numgen)};
    bch_encoded = step(bch_code, binary_string);
    bch_str = sprintf('%d',bch_encoded(1:3072))+";2";
    writelines(bch_str, ds, WriteMode="append");

    % Encode with RS code
    rs_code = rs_coders{rcodervar_r(numgen)};
    rs_encoded = step(rs_code,binary_string);
    rs_str = sprintf('%d',rs_encoded(1:3072))+";3";
    writelines(rs_str, ds, WriteMode="append");

    if mod(numgen,100) == 0
        fprintf('Iteration %d/%d\n',numgen,NumDatum)
    end
end


fclose('all');