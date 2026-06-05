# Script thuyet trinh TrafficVision-Transformer

Thoi luong goi y: 8-10 phut.  
Giong noi: binh tinh, ro rang, khong doc tung bullet tren slide.  
Thong diep chinh: day la nghien cuu tai hien co dieu chinh, tap trung vao pipeline va bien the nhe hoa phu hop tai nguyen sinh vien.

---

## Slide 1 - Tieu de

Kinh chao thay co va cac ban.

Nhom em xin trinh bay de tai TrafficVision-Transformer: tai hien co dieu chinh kien truc Spatio-Temporal Transformer nhe cho bai toan phat hien bat thuong giao thong trong video tu thiet bi bay khong nguoi lai.

Y tuong chinh cua nhom la: thay vi phat hien bat thuong bang nhan truc tiep, mo hinh se hoc quy luat cua cac doan giao thong binh thuong. Sau do, neu gap mot khung hinh ma mo hinh du doan kem, sai so lon, thi khung hinh do co kha nang la bat thuong.

Trong phan trinh bay nay, nhom se tap trung vao ba diem: vi sao bai toan kho, kien truc nhom da cai dat lai nhu the nao, va vi sao nhom dinh vi ket qua o muc prototype thay vi bao cao mot con so SOTA chua du tin cay.

Chuyen slide: Dau tien, nhom xin trinh bay boi canh va bai toan.

---

## Slide 2 - Noi dung trinh bay

Noi dung bai trinh bay gom sau phan.

Phan mot la mo dau va dat van de. Phan hai la dinh vi va dong gop cua nghien cuu. Phan ba la phuong phap de xuat, trong do trong tam la pipeline STE-TTE va Spatial Token Bottleneck. Phan bon la thuc nghiem va danh gia. Cuoi cung, nhom tong ket huong phat trien va phan cong cong viec.

Chuyen slide: Truoc het, tai sao nhom chon bai toan phat hien bat thuong giao thong tu video drone?

---

## Slide 3 - Mo dau va dat van de

Video drone co mot loi the rat lon trong giam sat giao thong: no cho goc nhin rong, linh hoat, va co the quan sat toan canh mot khu vuc do thi. Tuy nhien, chinh goc nhin nay cung tao ra nhieu kho khan cho bai toan thi giac may tinh.

Thu nhat, camera drone khong hoan toan co dinh. Video co the bi rung, bi thay doi goc nhin, hoac thay doi do cao. Thu hai, phuong tien trong anh thuong rat nho, dac biet khi quay tu tren cao. Thu ba, mat do giao thong thay doi manh theo tung canh va tung thoi diem.

Mot kho khan quan trong hon la du lieu bat thuong rat hiem. Cac su kien nhu tai nan, di sai lan, dung dot ngot, hay va cham khong xuat hien thuong xuyen. Neu dung hoc co giam sat, chung ta se can rat nhieu nhan bat thuong, nhung viec thu thap va gan nhan nhu vay ton cong va khong thuc te.

Vi vay, nhom chon huong hoc khong giam sat. Mo hinh chi hoc tren du lieu binh thuong. Cu the, nhom dung bai toan du doan khung hinh tuong lai: lay bon frame qua khu de du doan frame tiep theo. Neu canh giao thong dien ra binh thuong, mo hinh se du doan kha tot. Neu xuat hien hanh vi la, sai so du doan tang len, va sai so MSE nay duoc dung lam anomaly score.

Chuyen slide: Voi cach tiep can nay, dong gop cua nhom nam o dau?

---

## Slide 4 - Dinh vi va dong gop

Nhom khong dinh vi day la mot cong trinh phat minh hoan toan moi tu con so khong. Day la mot nghien cuu tai hien co dieu chinh tu huong Spatio-Temporal Transformer cho video giao thong drone.

Diem quan trong la nhom dua bai toan ve mot phien ban co the cai dat, chay thu, va kiem chung trong dieu kien tai nguyen han che cua sinh vien, cu the la moi truong Google Colab.

Dong gop thu nhat la cach nhe hoa mo hinh bang Spatial Token Bottleneck. Thay vi de temporal transformer xu ly tat ca patch token cua tat ca frame, nhom chi giu lai mot CLS token dai dien cho moi frame. Nhu vay, nhanh thoi gian chi can hoc tren chuoi token ngan hon rat nhieu.

Dong gop thu hai la nhom xay dung pipeline end-to-end: tu doc du lieu UIT-ADrone, tao sample gom 4 frame dau vao va 1 frame target, dua qua STE, TTE, decoder, tinh MSE, luu anomaly score va dua vao script tinh AUC-ROC.

Dong gop thu ba la phan tich tinh kha thi. Nhom khong co tinh bao cao mot ket qua huan luyen full-scale chua on dinh. Thay vao do, nhom trinh bay ro gioi han tinh toan va nhung phan da kiem chung duoc ve mat ky thuat.

Chuyen slide: Tiep theo la kien truc tong quan cua pipeline.

---

## Slide 5 - Phuong phap: Tong quan kien truc

Day la pipeline tong quan cua TrafficVision-Transformer.

Dau vao cua mo hinh la mot input clip gom bon khung hinh qua khu, ky hieu tu X t tru 4 den X t tru 1. Data Loader se tao mau gom 5 frame lien tiep: 4 frame dau dung lam input, frame cuoi la target de tinh loss.

Trong project, du lieu duoc tao thanh hai nhanh do phan giai. Nhanh standard, vi du 384 x 384, duoc dua vao encoder. Nhanh 256 x 256 duoc dung lam frame muc tieu cho decoder, giup giam bo nho khi tinh reconstruction loss.

Khoi dau tien la Spatial Transformer Encoder, hay STE. Moi frame duoc dua qua mot Vision Transformer de trich xuat dac trung khong gian. Diem nhom tap trung la token CLS cua tung frame, vi no dong vai tro nhu bieu dien tong hop cua frame do.

Sau STE la Spatial Token Bottleneck. Thay vi giu toan bo patch token, nhom chi giu mot CLS token cho moi frame. Voi 4 frame, ta co 4 token z1 den z4.

Chuoi token nay duoc dua vao Temporal Transformer Encoder, hay TTE. TTE them mot CLS token o dau chuoi, hoc quan he thoi gian giua cac frame, roi lay token dau ra lam bieu dien tong hop cua ca clip ngan.

Cuoi cung, Convolutional Decoder anh xa vector nay ve tensor dac trung, sau do upsample bang cac lop convolution de sinh anh du doan kich thuoc 256 x 256. Diem bat thuong duoc tinh bang MSE giua frame target that va frame du doan.

Noi ngan gon: pipeline nay hoc "binh thuong trong qua khu thi frame tiep theo nen trong nhu the nao". Khi thuc te khac nhieu so voi du doan, score tang.

Chuyen slide: Sau khi co kien truc, nhom thiet lap thuc nghiem nhu sau.

---

## Slide 6 - Thiet lap thuc nghiem

Bo du lieu nhom huong toi la UIT-ADrone, mot bo du lieu giao thong do thi quay tu drone, co nhan frame-level cho tap test. Day la bo du lieu phu hop voi bai toan cua nhom vi no co dung ngu canh giao thong tren khong, va co the danh gia bang AUC-ROC.

Ve cau hinh, moi sample gom 4 frame dau vao va 1 frame du doan. Anh vao encoder co kich thuoc 384, target reconstruction co kich thuoc 256 x 256. Khoi khong gian dua tren ViT-base patch16 trong ban cau hinh chinh, con Temporal Encoder duoc giu nong, voi so layer it hon, de giam chi phi huan luyen.

Optimizer su dung SGD voi momentum 0.9 va learning rate 10 mu tru 4. Loss la MSE reconstruction. Metric danh gia mong muon la AUC-ROC frame-level.

Nhom cung giu mot baseline CA trong project. Baseline nay cung theo huong du doan frame tuong lai, nhung cach ket hop dac trung khong gian - thoi gian khac voi STE-TTE. Vai tro cua baseline la lam diem doi chieu ve mat kien truc khi co du tai nguyen huan luyen day du.

Chuyen slide: Tuy nhien, phan ket qua cua nhom can duoc hieu dung la ket qua kiem chung prototype.

---

## Slide 7 - Ket qua kiem chung prototype

Day la slide quan trong, vi nhom muon trinh bay trung thuc ve ket qua.

Voi gioi han GPU tren Colab, dac biet khi train video transformer tren nhieu frame va doc du lieu tu Google Drive, viec huan luyen full-scale de lay con so AUC on dinh la rat ton thoi gian. Neu co bao cao mot con so chua hoi tu, no se khong phan anh dung nang luc cua mo hinh.

Vi vay, nhom uu tien kiem chung tinh dung dan cua pipeline.

Cu the, DataLoader da load dung mau gom 4 input frame va 1 target frame, dong thoi xu ly duoc hai nhanh do phan giai. Forward pass da duoc kiem tra: tensor di qua Spatial Transformer Encoder, qua Bottleneck, qua Temporal Transformer Encoder, va decoder sinh ra anh co shape B, 3, 256, 256.

Trong qua trinh train thu, loss MSE co the tinh va cap nhat. Checkpoint co the luu va tai lai cho inference. Sau inference, pipeline luu duoc chuoi anomaly score duoi dang file .npy, va cac file nay duoc dua vao script compute_auc.py de tinh AUC-ROC voi nhan frame-level.

Ket luan cua slide nay la: nhom da chung minh duoc phien ban kien truc nhe hoat dong dung ve mat ky thuat, va pipeline da san sang scale-up neu co GPU manh hon hoac thoi gian train dai hon.

Chuyen slide: Tu ket qua prototype do, nhom rut ra mot so uu diem, han che va huong phat trien.

---

## Slide 8 - Tong ket va huong phat trien

Ve uu diem, cach tiep can nay la hoc khong giam sat, nen phu hop voi bai toan ma du lieu bat thuong hiem va kho gan nhan. Ngoai ra, reconstruction error bang MSE rat truc quan: neu mo hinh du doan frame tuong lai kem, ta co co so de xem frame do la bat thuong.

Thiet ke cua nhom cung linh hoat de cai dat va debug. Spatial Token Bottleneck giup giam so token dua vao Temporal Transformer, tu do lam pipeline nhe hon so voi viec attention tren toan bo patch token cua video.

Tuy nhien, phuong phap co han che. Mot token cho moi frame co the lam mat chi tiet nho, trong khi phuong tien trong video drone von da rat nho. Ngoai ra, MSE toan anh co the bi anh huong boi rung camera, thay doi anh sang, hoac chuyen dong nen, chu khong chi do bat thuong giao thong.

Huong phat trien cua nhom gom ba diem. Thu nhat, dung pretrained ViT va fine-tune mot phan encoder de rut ngan thoi gian hoi tu. Thu hai, bo sung optical flow hoac crop vung phuong tien de mo hinh tap trung vao chuyen dong quan trong thay vi nen anh. Thu ba, khi co tai nguyen tinh toan lon hon, huan luyen full-scale tren UIT-ADrone va bao cao day du cac chi so nhu AUC va EER.

Chuyen slide: Cuoi cung la phan cong cong viec trong nhom.

---

## Slide 9 - Phan cong cong viec

Trong qua trinh thuc hien, nhom chia cong viec theo ba nhanh chinh.

Ban Dang Quang Minh phu trach tien xu ly du lieu, kiem tra DataLoader, phan danh gia va AUC. San pham tu phan nay la module du lieu va script tinh AUC.

Ban Vu Hoang Dieu Linh phu trach cai dat va phan tich mo hinh STE-TTE nhe, kiem tra forward pass, train script va checkpoint. Day la phan lien quan truc tiep den pipeline kien truc chinh.

Ban Le Hong Phuong Chi phu trach baseline, tong hop ket qua, viet bao cao va slide. Phan nay giup bai co cau truc so sanh va trinh bay ro rang hon.

Ve mat ma nguon, project duoc to chuc theo cac thu muc data, models, configs, scripts va results. Cach to chuc nay giup tach ro tung phan cua mot du an deep learning: du lieu, mo hinh, cau hinh, huan luyen va danh gia.

Chuyen slide: Phan trinh bay cua nhom den day la ket thuc.

---

## Slide 10 - Cam on va Q&A

Nhom em xin cam on thay co va cac ban da lang nghe.

Nhom rat mong nhan duoc cau hoi va gop y, dac biet ve hai huong: mot la cach nhe hoa kien truc Transformer cho video drone, hai la cach danh gia thuc nghiem hop ly trong dieu kien tai nguyen tinh toan han che.

---

# Cau hoi du phong

## 1. Tinh moi cua de tai la gi?

Tinh moi cua nhom khong nam o viec phat minh toan bo nguyen ly moi, ma nam o viec tai hien co dieu chinh va nhe hoa pipeline cho dieu kien tai nguyen han che.

Cu the, nhom dung Spatial Token Bottleneck: moi frame chi giu mot CLS token dai dien truoc khi dua vao Temporal Transformer. Dieu nay lam giam chi phi attention theo thoi gian va giup pipeline co the kiem chung tren Colab. Ngoai ra, nhom cai dat day du end-to-end tu data loader den anomaly score va AUC.

## 2. Tai sao khong bao cao AUC cuoi cung?

Vi nhom chua huan luyen full-scale den hoi tu. Video transformer tren UIT-ADrone ton tai nguyen, moi sample gom nhieu frame, va I/O tu Google Drive tren Colab cung cham. Neu bao cao mot con so khi train chua du, con so do de gay hieu nham.

Vi vay nhom chon cach bao cao trung thuc: ket qua prototype, kiem chung pipeline, forward pass, loss, checkpoint, inference va anomaly score.

## 3. Neu chi giu 1 token/frame thi co mat thong tin khong?

Co. Day la trade-off chinh. Mot token/frame co the lam mat chi tiet cua phuong tien nho. Nhung doi lai, temporal encoder nhe hon rat nhieu va co the chay trong dieu kien han che. Trong huong phat trien, nhom co the giu nhieu token quan trong hon, hoac ket hop crop vung phuong tien de giam mat thong tin.

## 4. Tai sao dung MSE lam anomaly score?

Vi bai toan theo huong future frame prediction. Mo hinh hoc du doan frame tiep theo tu cac frame binh thuong. Khi gap bat thuong, frame that se khac voi frame mo hinh du doan, nen MSE tang. MSE don gian, de giai thich va phu hop voi thiet lap unsupervised.

## 5. Huong cai tien kha thi nhat la gi?

Huong kha thi nhat la dung pretrained ViT va fine-tune mot phan encoder. Cach nay giam thoi gian hoi tu so voi train tu dau. Sau do co the them optical flow hoac smoothing score theo thoi gian de giam nhieu do rung camera va thay doi anh sang.

