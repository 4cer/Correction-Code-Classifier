% Basic setup
fclose('all');
NumDatum = 10000;

% Create text sources
fs_root = ".\AO\SRC\";
ds = "C:\Users\leejo\Documents\VSCode\AOP2\AO_Projekt_1\dataset\dataset.csv";

file_handles = [
    fopen(fs_root + "dzieje-tristana-i-izoldy.txt")
    fopen(fs_root + "homer-odyseja.txt")
    fopen(fs_root + "pan-tadeusz.txt")
    fopen(fs_root + "przygody-tomka-sawyera.txt")
    fopen(fs_root + "quo-vadis.txt")
    fopen(fs_root + "syzyfowe-prace.txt")
    fopen(fs_root + "w-pustyni-i-w-puszczy.txt")
];

file_contents = {
    fgets(file_handles(1), inf)
    fgets(file_handles(2), inf)
    fgets(file_handles(3), inf)
    fgets(file_handles(4), inf)
    fgets(file_handles(5), inf)
    fgets(file_handles(6), inf)
    fgets(file_handles(7), inf)
};

file_maxstart = strlength(file_contents)-385;

fclose('all');

% Coders
bch_code = comm.BCHEncoder(255, 239,bchgenpoly(255,239),6);
rs_code = comm.RSEncoder('BitInput',true, 'CodewordLength',255, 'MessageLength',239, 'ShortMessageLength',6);


numgen = 1;
rindices = randi([1,7],1,NumDatum);
rstarts = rand(1,NumDatum);

while numgen <= NumDatum
    index = rindices(numgen);
    start = floor(file_maxstart(index) * rstarts(numgen));
    utf8_string = file_contents{index}(start:start+383);
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
    writelines(sprintf('%d',binary_string(1:3071))+";0", ds, WriteMode="append")

    % Encode with Hamming code
    % hamming_code = comm.HammingEncoder;
    hamming_encoded = encode(binary_string, 255, 247, 'hamming/binary');
    writelines(sprintf('%d',hamming_encoded(1:3071))+";1", ds, WriteMode="append");

    % Encode with BCH code
    bch_encoded = step(bch_code, binary_string);
    writelines(sprintf('%d',bch_encoded(1:3071))+";2", ds, WriteMode="append");

    % Encode with RS code
    rs_encoded = step(rs_code,binary_string);
    writelines(sprintf('%d',rs_encoded(1:3071))+";3", ds, WriteMode="append");
    if mod(numgen,100) == 0
        fprintf('Iteration %d/%d\n',numgen,NumDatum)
    end
end


fclose('all');