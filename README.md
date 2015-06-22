Torch implementation of DRAW: A Recurrent Neural Network For Image Generation http://arxiv.org/pdf/1502.04623.pdf. Watch Deep Learning Lecture 14: Karol Gregor on Variational Autoencoders and Image Generation https://www.youtube.com/watch?v=P78QYjWh5sM&list=PLE6Wd9FR--EfW8dtjAuPoTuPcqmOV53Fu&index=3

Run 'th draw_attention.lua' in Terminal.app, it generates 'x_prediction', which you can plot by running plot_results*.lua in zbs-torch (https://github.com/soumith/zbs-torch) with QLua-LuaJit interpreter selected from 'Project' tab. Adjust the running time of the script by changing:
1. n_data (the number of MNIST examples to train on)
2. number of iterations
3. n_z, dimension of the hidden layer z
4. rnn_size, dimension of h_dec and h_enc


draw_attention.lua works with 28x28 MNIST dataset. You can adjust it to other datasets by changing A, N and replacing number '28' everywhere in the script. I haven't done it but it is possible.

draw_no_attention*.lua scripts implement DRAW without attention.
In draw_attention_read.lua only read is attentive, while write is without attention.

draw_no_attention*.lua scripts print arrays in the end, which helps to quickly estimate the quality of the results without plotting


