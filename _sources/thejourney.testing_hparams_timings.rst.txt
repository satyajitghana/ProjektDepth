4. Testing Timings and Hyper Params
===================================

GitHub Link : `<https://github.com/satyajitghana/ProjektDepth/blob/master/notebooks/08_DepthModel_Experiments_Timings.ipynb>`_
Colab Link  : `<https://colab.research.google.com/github/satyajitghana/ProjektDepth/blob/master/notebooks/08_DepthModel_Experiments_Timings.ipynb>`_

GPU: Tesla P100

Playing with batch_size
***********************

``BATCH_SIZE = 16``

.. code-block:: none

    total time : 170.1182 s
    the model took : 154.6660 s i.e. 0.9092 % of total execution
    data loading took : 0.9110 s i.e. 0.0054 % of total execution
    others took : 13.0727 s i.e. 0.0768 % of total execution

``BATCH_SIZE = 32``

.. code-block:: none

    total time : 157.4680 s
    the model took : 149.0861 s i.e. 0.9468 % of total execution
    data loading took : 0.7408 s i.e. 0.0047 % of total execution
    others took : 6.7184 s i.e. 0.0427 % of total execution

``BATCH_SIZE = 64``

.. code-block:: none

    total time : 149.5570 s
    the model took : 144.6822 s i.e. 0.9674 % of total execution
    data loading took : 0.6090 s i.e. 0.0041 % of total execution
    others took : 3.5036 s i.e. 0.0234 % of total execution

``BATCH_SIZE = 128``

.. code-block:: none

    total time : 145.5745 s
    the model took : 142.2546 s i.e. 0.9772 % of total execution
    data loading took : 0.5900 s i.e. 0.0041 % of total execution
    others took : 1.8308 s i.e. 0.0126 % of total execution

``BATCH_SIZE = 256``

.. code-block:: none

    RuntimeError: CUDA out of memory. Tried to allocate 1.12 GiB (GPU 0; 15.90 GiB total capacity; 12.72 GiB already allocated; 1023.81 MiB free; 14.20 GiB reserved in total by PyTorch)

Testing Model Timings
*********************

.. code-block:: python

    model = ResUNet()
    model = model.to(device)
    lossfn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    meow = torch.utils.data.Subset(train_subset, range(0, len(train_subset)//1))
    meow_loader = torch.utils.data.DataLoader(meow, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    pbar = tqdm(meow_loader, dynamic_ncols=True)

    other_time = 0

    start = time()

    other_s = time()
    model.train()
    other_e = time()

    other_time += other_e - other_s

    data_load_time = 0
    model_time = 0

    meow_time = 0

    for batch_idx, data in enumerate(pbar):

        other_s = time()
        optimizer.zero_grad()
        other_e = time()

        other_time += other_e - other_s

        load_s = time()

        data['bg'] = data['bg'].to(device)
        data['fg_bg'] = data['fg_bg'].to(device)
        data['depth_fg_bg'] = data['depth_fg_bg'].to(device)
        data['fg_bg_mask'] = data['fg_bg_mask'].to(device)

        load_e = time()

        data_load_time += load_e - load_s

        model_s = time() # model start

        x = torch.cat([data['bg'], data['fg_bg']], dim=1)
        d_out, s_out = model(x)

        l1 = lossfn(d_out, data['depth_fg_bg'])
        l2 = lossfn(s_out, data['fg_bg_mask'])

        loss = 2*l1 + l2

        loss.backward()
        optimizer.step()
        model_e = time() # model end

        model_time += model_e - model_s

        other_s = time()

        pbar.set_description(desc=f'loss={loss.item():.10f} batch_id={batch_idx}')


        # del data # and this shit was taking 0.07% of mah time
        if batch_idx % 200 == 0:
            torch.cuda.empty_cache() # this shit takes 8% of my frikking time

        other_e = time()

        other_time += other_e - other_s
    end = time()

    print(f'total time : {end-start:.4f} s')
    print(f'the model took : {model_time:.4f} s i.e. {(model_time / (end - start)):.4f} % of total execution')
    print(f'data loading took : {data_load_time:.4f} s i.e. {(data_load_time / (end - start)):.4f} % of total execution')
    print(f'others took : {other_time:.4f} s i.e. {(other_time / (end - start)):.4f} % of total execution')



OUTPUT

.. code-block:: none

    total time : 2331.8488 s
    the model took : 2289.7168 s i.e. 0.9819 % of total execution
    data loading took : 10.2884 s i.e. 0.0044 % of total execution
    others took : 28.5874 s i.e. 0.0123 % of total execution